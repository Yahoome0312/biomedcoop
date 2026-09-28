"""Run 36 semantic-distillation experiments with lambda=0.1, temperature=0.5."""
import csv
import json
import statistics
from pathlib import Path
from scripts import run_three_methods as experiments

ROOT = experiments.ROOT
OUT = ROOT / 'output/text_guided_visual_semantic_distill_lambda0p1_tau0p5_seed123'
BASELINE = ROOT / 'output/class_text_token_replacement_layer7_seed123/class_text_token'
experiments.OUT = OUT
experiments.METHODS = ('semantic_distill',)
experiments.SHOTS = (4, 8, 16, 32)
experiments.GPUS = (0, 0, 1, 1, 2, 2, 6, 6, 7, 7)
experiments.CASES = [('semantic_distill', d, k, s) for d in experiments.DATASETS
                     for k in experiments.SHOTS for s in experiments.SEEDS]


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = []
    detailed = []
    for dataset in experiments.DATASETS:
        for shots in experiments.SHOTS:
            original, semantic = [], []
            for seed in experiments.SEEDS:
                rel = Path(dataset)/f'shots_{shots}'/f'seed{seed}'
                original.append(json.loads((BASELINE/rel/'test_metrics.json').read_text())['test_accuracy'])
                dest = OUT/'semantic_distill'/rel
                path = dest/'test_metrics.json'
                if path.exists():
                    result = json.loads(path.read_text())
                    semantic.append(result['test_accuracy'])
                    detailed.append(dict(dataset=dataset, shots=shots, seed=seed,
                        selected_epoch=result['selected_epoch'], original_accuracy=original[-1],
                        semantic_accuracy=result['test_accuracy'], delta=result['test_accuracy']-original[-1]))
                events = list(dest.glob('tensorboard/events.out.tfevents.*'))
                if not events:
                    events = list(dest.rglob('events.out.tfevents.*'))
                if events:
                    acc = EventAccumulator(str(events[0].parent), size_guidance={'scalars': 0})
                    acc.Reload()
                    tags = acc.Tags()['scalars']
                    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
                    curve_rows = []
                    for name, axis in [('loss_ce', axes[0]), ('loss_sem', axes[0]), ('semantic_grad_norm', axes[1])]:
                        tag = next((t for t in tags if t.endswith('/'+name)), None)
                        if tag:
                            values = acc.Scalars(tag)
                            axis.plot([v.step for v in values], [v.value for v in values], label=name)
                            curve_rows += [{'metric':name,'step':v.step,'value':v.value} for v in values]
                    for axis in axes:
                        axis.legend(); axis.set_xlabel('training step')
                    fig.tight_layout(); fig.savefig(dest/'semantic_curves.png'); plt.close(fig)
                    with (dest/'semantic_curves.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['metric','step','value'])
                        writer.writeheader(); writer.writerows(curve_rows)
            if len(semantic) == 3:
                rows.append(dict(dataset=dataset, shots=shots, original_mean=statistics.mean(original),
                    original_std=statistics.stdev(original), semantic_mean=statistics.mean(semantic),
                    semantic_std=statistics.stdev(semantic), delta=statistics.mean(semantic)-statistics.mean(original)))
    OUT.mkdir(parents=True, exist_ok=True)
    if detailed:
        with (OUT/"results_detailed.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed[0]))
            writer.writeheader(); writer.writerows(detailed)
    if rows:
        with (OUT/'comparison_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    setting_count = len(experiments.DATASETS) * len(experiments.SHOTS)
    lines = [f'完成 {len(rows)}/{setting_count} setting；每个 setting 为 seed 1/2/3，std 为样本标准差。',
             '', '|Setting|Original|Semantic Distill|Δ (pp)|', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['dataset']} {r['shots']}-shot|{r['original_mean']:.4f} ± {r['original_std']:.4f}|{r['semantic_mean']:.4f} ± {r['semantic_std']:.4f}|{r['delta']:+.4f}|")
    if len(rows) == setting_count:
        wins = sum(r['delta'] > 1e-8 for r in rows)
        ties = sum(abs(r['delta']) <= 1e-8 for r in rows)
        lines += ['', f"{setting_count} setting 等权平均：Original {statistics.mean(r['original_mean'] for r in rows):.4f}%；Semantic {statistics.mean(r['semantic_mean'] for r in rows):.4f}%。",
                  f"win/tie/loss：{wins}/{ties}/{setting_count-wins-ties}；最大单 setting 下降：{max(0, -min(r['delta'] for r in rows)):.4f} pp。"]
    lines += ['', f'Original 来源：{BASELINE}；layer7、FUSION_WEIGHT=1.0、Mean-50、相同优化器/训练配置/seed，按 validation accuracy 选模后独立 test。',
              '固定 WEIGHT=0.1、TEMPERATURE=0.5，无温度平方因子。各运行目录输出 semantic_curves.csv/png 和 semantic_gradient_audit.json。']
    (OUT/'comparison_report.md').write_text('\n'.join(lines))

experiments.summarize = summarize
if __name__ == '__main__':
    experiments.write_json(OUT/'_manager/plan.json', dict(
        cases=experiments.CASES, gpus=experiments.GPUS, baseline=str(BASELINE),
        weight=0.1, temperature=0.5, insert_layer=7, fusion_weight=1.0))
    experiments.main()
