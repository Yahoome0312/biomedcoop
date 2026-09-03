"""Build support-only frozen-BiomedCLIP soft confusion probability Banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
DASSL = REPO / "Dassl.pytorch"
for location in (REPO, DASSL):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import train
from dassl.data.data_manager import DatasetWrapper
from dassl.data.datasets import build_dataset
from dassl.data.transforms import build_transform
from dassl.utils import set_random_seed
from models.biomedclip_loader import BIOMEDCLIP_MODEL_ID, load_biomedclip
from models.confusion_aware import (
    PRIOR_TYPE,
    bank_file,
    compute_hard_confusion_counts,
    compute_soft_confusion_prior,
    save_soft_confusion_bank,
    support_fingerprint,
    support_records,
)
from open_clip.src.open_clip import get_tokenizer


TEMPLATE = "a photo of a {}."


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _cfg(args, shots, seed):
    return train.setup_cfg(
        Namespace(
            root=str(args.data_root),
            output_dir=str(args.output_root / "_bank_build_logs"),
            resume="",
            seed=int(seed),
            source_domains=None,
            target_domains=None,
            transforms=None,
            trainer="CoOpVPT_BiomedCLIP",
            backbone="",
            head="",
            dataset_config_file=str(REPO / "configs/datasets/dermamnist.yaml"),
            config_file=str(
                REPO
                / "configs/trainers/CoOp"
                / "dermamnist_native_vpt_multitext_tcp.yaml"
            ),
            eval_only=False,
            model_dir="",
            load_epoch=None,
            no_train=False,
            opts=[
                "DATASET.NUM_SHOTS",
                str(shots),
            ],
        )
    )


def _preprocess_fingerprint(cfg):
    value = {
        "size": list(cfg.INPUT.SIZE),
        "interpolation": cfg.INPUT.INTERPOLATION,
        "pixel_mean": list(cfg.INPUT.PIXEL_MEAN),
        "pixel_std": list(cfg.INPUT.PIXEL_STD),
        "transform": "build_transform(is_train=False)",
    }
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


@torch.no_grad()
def build_one(args, shots, seed):
    cfg = _cfg(args, shots, seed)
    set_random_seed(int(seed))
    dataset = build_dataset(cfg)
    classnames = list(dataset.classnames)
    support = list(dataset.train_x)
    records = support_records(support)

    support_paths = {str(Path(item.impath).resolve()) for item in support}
    held_out_paths = {
        str(Path(item.impath).resolve()) for item in list(dataset.val) + list(dataset.test)
    }
    overlap = support_paths & held_out_paths
    if overlap:
        raise RuntimeError("Support overlaps val/test: {}".format(sorted(overlap)[:5]))

    transform = build_transform(cfg, is_train=False)
    wrapper = DatasetWrapper(cfg, support, transform=transform, is_train=False)
    loader = DataLoader(
        wrapper,
        batch_size=train.FIXED_BATCH_SIZE,
        shuffle=False,
        num_workers=train.FIXED_NUM_WORKERS,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
    model, _ = load_biomedclip(vpt_enabled=False)
    model = model.to(device).eval()
    tokenizer = get_tokenizer(BIOMEDCLIP_MODEL_ID)
    prompts = [TEMPLATE.format(name.replace("_", " ")) for name in classnames]
    tokenized = torch.cat([tokenizer(prompt) for prompt in prompts]).to(device)
    text_features = model.encode_text(tokenized, normalize=True).float()
    logit_scale = model.logit_scale.exp().detach().float()

    probabilities = []
    labels = []
    offset = 0
    for batch in loader:
        images = batch["img"].to(device)
        image_features = model.encode_image(images, normalize=True).float()
        logits = logit_scale * image_features @ text_features.t()
        batch_probabilities = logits.softmax(dim=-1).cpu()
        batch_labels = batch["label"].long().cpu()
        probabilities.append(batch_probabilities)
        labels.append(batch_labels)
        for index in range(batch_probabilities.shape[0]):
            row = batch_probabilities[index]
            prediction = int(row.argmax().item())
            records[offset + index].update(
                frozen_predicted_label=prediction,
                frozen_confidence=float(row[prediction].item()),
                frozen_class_probabilities=[float(value) for value in row.tolist()],
            )
        offset += batch_probabilities.shape[0]

    probabilities = torch.cat(probabilities).float()
    labels = torch.cat(labels).long()
    soft_prior, class_counts = compute_soft_confusion_prior(
        probabilities, labels, len(classnames)
    )
    hard_counts = compute_hard_confusion_counts(
        probabilities, labels, len(classnames)
    )
    if not torch.equal(class_counts, torch.bincount(labels, minlength=len(classnames))):
        raise RuntimeError("Support class counts are inconsistent")
    bank_fingerprint = _sha256_bytes(soft_prior.contiguous().numpy().tobytes())
    metadata = {
        "schema_version": 1,
        "prior_type": PRIOR_TYPE,
        "dataset": cfg.DATASET.NAME,
        "shots": int(shots),
        "seed": int(seed),
        "class_order": classnames,
        "support_size_per_class": class_counts.tolist(),
        "support_fingerprint": support_fingerprint(support),
        "bank_fingerprint": bank_fingerprint,
        "model_id": BIOMEDCLIP_MODEL_ID,
        "template": TEMPLATE,
        "preprocess_fingerprint": _preprocess_fingerprint(cfg),
        "support_only": True,
        "val_test_images_encoded": False,
        "off_diagonal_renormalized": False,
    }
    payload = {
        "metadata": metadata,
        "soft_prior": soft_prior,
        "hard_confusion_counts": hard_counts,
        "support_records": records,
    }
    path = bank_file(args.output_root, cfg.DATASET.NAME, shots, seed)
    save_soft_confusion_bank(path, payload)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "soft_prior": soft_prior.tolist(),
                "hard_confusion_counts": hard_counts.tolist(),
                "support_records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("WROTE {}".format(path), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,default=Path("/mnt/nas1/disk09/yuejianwu/biomedcoop/data"),)
    parser.add_argument(
        "--output-root", type=Path, default=REPO / "output" / "soft_confusion_banks"
    )
    parser.add_argument("--shots", nargs="+", type=int, default=(4, 8, 16, 32))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    for shots in args.shots:
        for seed in train.EXPERIMENT_SEEDS:
            build_one(args, shots, seed)


if __name__ == "__main__":
    main()
