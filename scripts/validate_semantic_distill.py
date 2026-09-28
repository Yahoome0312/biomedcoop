"""Real-backbone smoke check before releasing reserved GPUs."""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import train
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]

def main():
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"
    for dataset in ('DermaMNIST', 'Kvasir', 'CHMNIST'):
        args = SimpleNamespace(root=str(ROOT/'data'), output_dir=str(ROOT/'output/semantic_distill_validation'/dataset),
            resume='', source_domains=None, target_domains=None, transforms=None,
            trainer='CoOpVPT_BiomedCLIP', backbone='', head='', seed=1,
            dataset_config_file=str(ROOT/'configs/datasets'/f'{dataset.lower()}.yaml'),
            config_file=str(ROOT/'configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml'),
            opts=['DATASET.NUM_SHOTS','4','DATALOADER.TRAIN_X.BATCH_SIZE','2',
                  'DATALOADER.NUM_WORKERS','0','TRAINER.SEMANTIC_DISTILL.ENABLED','True'])
        cfg = train.setup_cfg(args)
        set_random_seed(1)
        trainer = build_trainer(cfg)
        trainer.batch_idx = 0
        trainer.num_batches = len(trainer.train_loader_x)
        trainer.epoch = 0
        summary = trainer.forward_backward(next(iter(trainer.train_loader_x)))
        assert all(torch.isfinite(torch.tensor(v)) for v in summary.values())
        trainer.set_model_mode('eval')
        images, labels = trainer.parse_batch_train(next(iter(trainer.train_loader_x)))
        with torch.no_grad():
            logits = trainer.model(images)
            trainer.cfg.defrost()
            trainer.cfg.TRAINER.SEMANTIC_DISTILL.ENABLED = False
            trainer.cfg.freeze()
            baseline, losses = trainer._compute_training_loss(images, labels)
            assert set(losses) == {'loss', 'loss_ce'}
            torch.testing.assert_close(logits, baseline, rtol=0, atol=0)
        trainer.cfg.defrost()
        trainer.cfg.TRAINER.SEMANTIC_DISTILL.ENABLED = True
        trainer.cfg.freeze()
        trainer.save_model(0, cfg.OUTPUT_DIR)
        trainer.load_model(cfg.OUTPUT_DIR, epoch=1)
        (Path(cfg.OUTPUT_DIR)/'validation.json').write_text(json.dumps(summary, indent=2))
        print(f'VALIDATION PASSED {dataset}: {summary}', flush=True)
        del trainer
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
