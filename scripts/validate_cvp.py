"""Real BiomedCLIP CVP smoke and unchanged batch32 resource check."""
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import train
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/class_conditioned_visual_prompt_text0p5_visual0p5_validation'


def main():
    os.environ['TMPDIR'] = '/tmp'
    tempfile.tempdir = '/tmp'
    records = []
    for dataset in ('DermaMNIST', 'Kvasir', 'CHMNIST'):
        args = SimpleNamespace(root=str(ROOT / 'data'), output_dir=str(OUT / dataset),
            resume='', source_domains=None, target_domains=None, transforms=None,
            trainer='CoOpVPT_BiomedCLIP', backbone='', head='', seed=1,
            dataset_config_file=str(ROOT / 'configs/datasets' / f'{dataset.lower()}.yaml'),
            config_file=str(ROOT / 'configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml'),
            opts=['DATASET.NUM_SHOTS', '32', 'TRAINER.CVP.ENABLED', 'True',
                  'TRAINER.CVP.INSERT_LAYER', '7', 'TRAINER.CVP.NUM_TOKENS', '4',
                  'TRAINER.CVP.BOTTLENECK_DIM', '128', 'TRAINER.CVP.FUSION_WEIGHT', '0.5', 'TRAINER.TCP.ENABLED', 'True',
                  'TRAINER.TCP.INSERT_LAYER', '7', 'TRAINER.TCP.FUSION_WEIGHT', '0.5'])
        cfg = train.setup_cfg(args)
        # Only the validation loader avoids worker startup; formal batches remain 32.
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.freeze()
        set_random_seed(1)
        trainer = build_trainer(cfg)
        trainer.batch_idx = 0
        trainer.num_batches = len(trainer.train_loader_x)
        trainer.epoch = 0
        batch = next(iter(trainer.train_loader_x))
        assert len(batch['img']) == 32
        small = {key: value[:2] for key, value in batch.items()}
        trainer.set_model_mode('train')
        summary = trainer.forward_backward(small)
        assert set(summary) == {'loss', 'loss_ce', 'acc', 'lr'}
        assert all(torch.isfinite(torch.tensor(v)) for v in summary.values())
        model = trainer._unwrapped_model()
        gradient_norms = {name: float(p.grad.float().norm()) for name, p in model.cvp.named_parameters()}
        assert all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient_norms.values())
        visual_gradient = model.image_encoder.visual_prompt.prompt_embeddings.grad
        assert visual_gradient is not None and torch.isfinite(visual_gradient).all()
        assert visual_gradient[7].norm() > 0 and visual_gradient[8:].count_nonzero() == 0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        trainer.set_model_mode('eval')
        images, labels = trainer.parse_batch_train(small)
        with torch.no_grad():
            tokens = model.cvp(model.text_encoder.class_prior)
            logits, text, image = model(images, return_features=True)
            c = len(trainer.dm.dataset.classnames)
            assert tokens.shape == (c, 4, 768)
            assert text.shape == (c, 512) and image.shape == (2, c, 512)
            assert logits.shape == (2, c)
            output1, losses = trainer._compute_training_loss(images, labels)
            output2, _ = trainer._compute_training_loss(images, (labels + 1) % c)
            assert set(losses) == {'loss', 'loss_ce'}
            torch.testing.assert_close(output1, output2, rtol=0, atol=0)
        trainer.save_model(0, cfg.OUTPUT_DIR)
        checkpoint = trainer.load_model(cfg.OUTPUT_DIR, epoch=1)
        assert checkpoint['cvp_enabled']
        assert checkpoint['cvp_fusion_weight'] == checkpoint['fusion_weight'] == .5
        with torch.no_grad():
            torch.testing.assert_close(logits, model(images), rtol=0, atol=0)
        trainer.set_model_mode('train')
        trainer.model_zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        formal_summary = trainer.forward_backward(batch)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        assert all(torch.isfinite(torch.tensor(v)) for v in formal_summary.values())
        record = dict(cvp_fusion_weight=0.5, dataset=dataset, classes=c, small_batch=2, formal_batch=32,
            tokens=list(tokens.shape), text=list(text.shape), conditioned_images=list(image.shape),
            logits=list(logits.shape), cvp_gradient_norms=gradient_norms,
            visual_layer7_gradient_norm=float(visual_gradient[7].float().norm()), smoke_loss=summary,
            batch32_loss=formal_summary, batch32_seconds=elapsed,
            batch32_peak_mib=torch.cuda.max_memory_allocated() / 1024**2,
            parameter_counts=trainer._parameter_count_manifest())
        (Path(cfg.OUTPUT_DIR) / 'validation.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        print(f'CVP VALIDATION PASSED: {json.dumps(record)}', flush=True)
        del checkpoint, logits, output1, output2, losses, tokens, text, image, model, trainer
        torch.cuda.empty_cache()
    (OUT / 'validation_summary.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
