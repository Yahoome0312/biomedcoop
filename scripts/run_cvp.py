"""Run 36 independent CVP+CE experiments with text/visual fusion=0.5."""
import csv
import argparse
import ctypes
import json
import os
import queue
import select
import signal
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

from scripts import run_three_methods as experiments

OUT = experiments.ROOT / "output/class_conditioned_visual_prompt_text0p5_visual0p5_seed123"
BASELINE = experiments.ROOT / "output/class_text_token_fusion_0p5_layers6_11_seed123/layer_7"
VALIDATION = experiments.ROOT / "output/class_conditioned_visual_prompt_text0p5_visual0p5_validation/validation_summary.json"
_run_case = experiments.run_case
VISUAL_FUSION_WEIGHT = 0.5
MEMORY_REQUIRED = {}
GPU_LOCKS = defaultdict(threading.Lock)
RESERVATIONS = defaultdict(dict)


def gpu_capacity(gpu):
    """Free memory plus memory already owned by this CVP output directory."""
    available = int(subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True).strip())
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"
    ], text=True)
    for row in processes.splitlines():
        pid, memory = row.split(",")
        try:
            command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode()
        except FileNotFoundError:
            continue
        if str(OUT / "cvp") in command:
            available += int(memory)
    return available


def run_case(gpu, case):
    """Wait for measured CVP memory capacity without disturbing other jobs."""
    if (experiments.run_dir(case) / "test_metrics.json").exists():
        return "already_complete"
    required = MEMORY_REQUIRED[case[1]]
    while True:
        with GPU_LOCKS[gpu]:
            available = gpu_capacity(gpu) - sum(RESERVATIONS[gpu].values())
            if available >= required:
                RESERVATIONS[gpu][case] = required
                break
        print(f"GPU{gpu}: waiting for {required:.0f} MiB; free {available} MiB; case={case}", flush=True)
        time.sleep(900)
    try:
        return _run_case(gpu, case, cvp_fusion_weight=VISUAL_FUSION_WEIGHT)
    finally:
        with GPU_LOCKS[gpu]:
            RESERVATIONS[gpu].pop(case)


def adopt_children(manager, jobs_per_gpu):
    command = Path(f"/proc/{manager}/cmdline").read_bytes().decode()
    if "scripts.run_cvp" not in command:
        raise RuntimeError("Only a CVP queue manager may be adopted")
    os.kill(manager, signal.SIGSTOP)
    active = defaultdict(list)
    try:
        while Path(f"/proc/{manager}/stat").read_text().rsplit(')', 1)[1].split()[0] not in {'T', 't'}:
            time.sleep(0.01)
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(stat[1]) != manager or stat[0] == 'Z':
                    continue
                args = (proc / 'cmdline').read_bytes().decode().split('\0')
                flag = '--output-dir' if '--output-dir' in args else '--run-dir'
                if flag not in args:
                    continue
                dest = Path(args[args.index(flag) + 1])
                method, dataset, shots, seed = dest.relative_to(OUT).parts
                case = (method, dataset, int(shots.removeprefix('shots_')), int(seed.removeprefix('seed')))
                env = dict(v.split('=', 1) for v in (proc / 'environ').read_bytes().decode().split('\0') if '=' in v)
                gpu = int(env['CUDA_VISIBLE_DEVICES'])
                pidfd = ctypes.CDLL(None, use_errno=True).syscall(434, int(proc.name), 0)
                if pidfd < 0:
                    raise OSError(ctypes.get_errno(), "Cannot wait for adopted CVP process")
                active[gpu].append((case, int(proc.name), pidfd))
                RESERVATIONS[gpu][case] = MEMORY_REQUIRED[dataset]
            except FileNotFoundError:
                continue
        if any(gpu not in experiments.GPUS or len(items) > jobs_per_gpu
               for gpu, items in active.items()):
            raise RuntimeError("Adopted CVP tasks exceed configured GPUs or slots")
        experiments.write_json(OUT / '_manager/handoff.json', dict(
            previous_manager=manager, active=[dict(gpu=gpu, case=case, pid=pid)
                for gpu, items in active.items() for case, pid, _ in items]))
    except Exception:
        for gpu, items in active.items():
            for case, _, pidfd in items:
                os.close(pidfd)
                RESERVATIONS[gpu].pop(case, None)
        try:
            os.kill(manager, signal.SIGCONT)
        except ProcessLookupError:
            pass
        raise
    return active


def run_queue(jobs_per_gpu, previous_manager=None):
    active = adopt_children(previous_manager, jobs_per_gpu) if previous_manager else {}
    adopted = {item[0] for items in active.values() for item in items}
    pending = queue.Queue()
    for case in experiments.CASES:
        if case not in adopted:
            pending.put(case)
    records = []
    lock = threading.Lock()

    def process(gpu, case):
        try:
            status = run_case(gpu, case)
            record = dict(case=case, gpu=gpu, status=status)
        except Exception as exc:
            record = dict(case=case, gpu=gpu, status='failed', error=str(exc))
        with lock:
            records.append(record)
            experiments.write_json(OUT / '_manager/status.json', dict(
                total=len(experiments.CASES), processed=len(records),
                completed=sum(r['status'] != 'failed' for r in records), records=records))
            summarize()
            print(json.dumps(record), flush=True)

    def worker(gpu, slot):
        inherited = active.get(gpu, [])
        if slot < len(inherited):
            case, pid, pidfd = inherited[slot]
            print(f'GPU{gpu} slot{slot}: preserving existing PID{pid}, case={case}', flush=True)
            select.select([pidfd], [], [])
            os.close(pidfd)
            with GPU_LOCKS[gpu]:
                RESERVATIONS[gpu].pop(case)
            process(gpu, case)
        while True:
            try:
                case = pending.get_nowait()
            except queue.Empty:
                return
            process(gpu, case)

    threads = [threading.Thread(target=worker, args=(gpu, slot))
               for gpu in experiments.GPUS for slot in range(jobs_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if previous_manager:
        os.kill(previous_manager, signal.SIGTERM)
        os.kill(previous_manager, signal.SIGCONT)


def summarize():
    detailed, summary = [], []
    for dataset in experiments.DATASETS:
        for shots in experiments.SHOTS:
            rows = []
            for seed in experiments.SEEDS:
                rel = Path(dataset) / f"shots_{shots}" / f"seed{seed}"
                path = OUT / "cvp" / rel / "test_metrics.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                original = json.loads((BASELINE / rel / "test_metrics.json").read_text())
                row = dict(dataset=dataset, shots=shots, seed=seed,
                           selected_epoch=result["selected_epoch"],
                           validation_accuracy=result["validation_accuracy"],
                           original_accuracy=original["test_accuracy"],
                           cvp_accuracy=result["test_accuracy"],
                           delta=result["test_accuracy"] - original["test_accuracy"])
                row.update({field: result[field] for field in (
                    "cvp_enabled", "cvp_insert_layer", "cvp_num_tokens", "cvp_bottleneck_dim", "fusion_weight", "cvp_fusion_weight")})
                if (row["cvp_enabled"], row["cvp_insert_layer"], row["cvp_num_tokens"],
                    row["cvp_bottleneck_dim"], row["fusion_weight"], row["cvp_fusion_weight"]) != (True, 7, 4, 128, 0.5, VISUAL_FUSION_WEIGHT):
                    raise RuntimeError(f"Unexpected checkpoint CVP settings: {path}")
                detailed.append(row)
                rows.append(row)
            if len(rows) == 3:
                summary.append(dict(dataset=dataset, shots=shots,
                    original_mean=statistics.mean(r["original_accuracy"] for r in rows),
                    original_std=statistics.stdev(r["original_accuracy"] for r in rows),
                    cvp_mean=statistics.mean(r["cvp_accuracy"] for r in rows),
                    cvp_std=statistics.stdev(r["cvp_accuracy"] for r in rows),
                    delta=statistics.mean(r["delta"] for r in rows)))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("results_detailed.csv", detailed), ("comparison_summary.csv", summary)):
        if rows:
            with (OUT / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = [f"完成 {len(detailed)}/36 组、{len(summary)}/12 setting；std 为样本标准差。", "",
             "|Setting|Original|CVP|Δ (pp)|", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"|{row['dataset']} {row['shots']}-shot|{row['original_mean']:.4f} ± {row['original_std']:.4f}|"
                     f"{row['cvp_mean']:.4f} ± {row['cvp_std']:.4f}|{row['delta']:+.4f}|")
    if len(summary) == 12:
        wins = sum(r["delta"] > 1e-8 for r in summary)
        ties = sum(abs(r["delta"]) <= 1e-8 for r in summary)
        lines += ["", f"12-setting等权平均：Original {statistics.mean(r['original_mean'] for r in summary):.4f}%；"
                  f"CVP {statistics.mean(r['cvp_mean'] for r in summary):.4f}%。",
                  f"win/tie/loss：{wins}/{ties}/{12-wins-ties}。"]
    lines += ["", f"对照：{BASELINE}；两端代码layer7（第8个Block），文本fusion0.5，视觉0.5融合，仅CE。",
              "每组按validation accuracy选模后独立test，训练和测试遍历全部候选类别。"]
    (OUT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(jobs_per_gpu=1, previous_manager=None):
    if jobs_per_gpu not in (1, 2):
        raise ValueError("CVP supports one or two jobs per GPU")
    validations = json.loads(VALIDATION.read_text())
    if {r["dataset"] for r in validations if r["formal_batch"] == 32 and
            r.get("cvp_fusion_weight") == VISUAL_FUSION_WEIGHT} != set(experiments.DATASETS):
        raise RuntimeError("All three real-backbone CVP batch32 checks must pass before training")
    MEMORY_REQUIRED.update({r["dataset"]: r["batch32_peak_mib"] + 2048 for r in validations})
    experiments.OUT = OUT
    experiments.METHODS = ("cvp",)
    experiments.GPUS = (1, 2, 6, 7)
    experiments.CASES = [("cvp", d, k, s) for d in experiments.DATASETS
                         for k in experiments.SHOTS for s in experiments.SEEDS]
    experiments.summarize = summarize
    experiments.run_case = run_case
    experiments.write_json(OUT / "_manager/plan.json", dict(
        cases=experiments.CASES, gpus=experiments.GPUS, baseline=str(BASELINE),
        cvp_enabled=True, cvp_insert_layer=7, cvp_num_tokens=4,
        cvp_bottleneck_dim=128, fusion_weight=0.5, cvp_fusion_weight=VISUAL_FUSION_WEIGHT, semantic_enabled=False,
        jobs_per_gpu=jobs_per_gpu))
    if jobs_per_gpu == 1 and previous_manager is None:
        experiments.main()
    else:
        run_queue(jobs_per_gpu, previous_manager)
    results = [OUT / "cvp" / d / f"shots_{k}" / f"seed{s}" / "test_metrics.json"
               for _, d, k, s in experiments.CASES]
    if any(not path.exists() for path in results):
        raise RuntimeError("CVP queue finished with incomplete experiments; inspect _manager/status.json")
    import torch
    for path in results:
        metrics = json.loads(path.read_text())
        dest = path.parent
        selected = json.loads((dest / "best_validation_accuracy.json").read_text())
        if (metrics["selected_epoch"] != selected["epoch"] or
                metrics["validation_accuracy"] != selected["selection_value"] or
                not (dest / "prompt_parameters/model-best.pth.tar").exists()):
            raise RuntimeError(f"CVP final checkpoint selection mismatch: {dest}")
        checkpoint = torch.load(dest / "prompt_parameters/model-best.pth.tar",
                                map_location="cpu", weights_only=False)
        fields = ("cvp_enabled", "cvp_insert_layer", "cvp_num_tokens", "cvp_bottleneck_dim", "fusion_weight", "cvp_fusion_weight")
        if checkpoint["epoch"] != selected["epoch"] or any(
                checkpoint[field] != metrics[field] for field in fields):
            raise RuntimeError(f"CVP saved checkpoint metadata mismatch: {dest}")
        if checkpoint["semantic_distill"]["ENABLED"]:
            raise RuntimeError(f"CVP checkpoint unexpectedly enabled Semantic Distill: {dest}")
    summarize()
    experiments.write_json(OUT / "_manager/final_validation.json", dict(
        completed=36, settings=12, checkpoint_selection_verified=True,
        checkpoint_cvp_metadata_verified=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs-per-gpu', type=int, default=1, choices=(1, 2))
    parser.add_argument('--adopt-manager', type=int, default=None)
    args = parser.parse_args()
    main(args.jobs_per_gpu, args.adopt_manager)
