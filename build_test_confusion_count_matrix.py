"""Build a hard confusion-count matrix from the test split."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dassl.config import get_cfg_default
from open_clip.src.open_clip import get_tokenizer

from datasets import build_dataset
from models.biomedclip_loader import BIOMEDCLIP_MODEL_ID, load_biomedclip
from models.original_style_tcp import (
    DESCRIPTION_COUNT,
    build_frozen_description_bank,
)
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


BATCH_SIZE = 32


class TestImages(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(item.impath) as image:
            image = image.convert("RGB")
        return self.transform(image), item.label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a hard confusion-count matrix from the test split"
    )
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument(
        "--dataset-config-file", required=True, help="dataset YAML config"
    )
    parser.add_argument("--output", default=".", help="output directory")
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg_default()
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.merge_from_file(args.dataset_config_file)
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = -1
    cfg.freeze()
    return cfg


def save_figure(matrix, classnames, path):
    size = max(7, 0.8 * len(classnames))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix.numpy(), cmap="Blues", vmin=0)
    axis.set_xticks(range(len(classnames)), classnames, rotation=45, ha="right")
    axis.set_yticks(range(len(classnames)), classnames)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Test-set Confusion Matrix (Counts)")
    threshold = matrix.max().item() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column].item()
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    dataset = build_dataset(build_cfg(args))
    classnames = dataset.classnames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = load_biomedclip()
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tokenizer = get_tokenizer(BIOMEDCLIP_MODEL_ID)
    description_bank, _ = build_frozen_description_bank(
        model,
        tokenizer,
        classnames,
        BIOMEDCOOP_TEMPLATES,
        expected_count=DESCRIPTION_COUNT,
        batch_size=BATCH_SIZE,
    )
    class_text_features = F.normalize(description_bank.mean(dim=1), dim=-1).to(
        device
    )

    loader = DataLoader(
        TestImages(dataset.test, preprocess),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    num_classes = len(classnames)
    matrix = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=device
    )

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            image_features = model.encode_image(images, normalize=True).float()
            predictions = (image_features @ class_text_features.T).argmax(dim=-1)
            indices = labels * num_classes + predictions
            matrix += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    matrix = matrix.cpu()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "test_confusion_count_matrix.pt"
    figure_path = output_dir / "test_confusion_count_matrix.png"
    torch.save(matrix, matrix_path)
    save_figure(matrix, classnames, figure_path)

    print("Class names:")
    for index, classname in enumerate(classnames):
        print(f"  {index}: {classname}")
    torch.set_printoptions(profile="full", linewidth=200)
    print("Test confusion count matrix [true, predicted]:")
    print(matrix)
    print(f"Evaluated {matrix.sum().item()} test images")
    print(f"Saved tensor to {matrix_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
