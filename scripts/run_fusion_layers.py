"""Compare TCP fusion insertion at the remaining blocks in BERT's latter half."""

import csv
import json
import queue
import statistics
import threading

from scripts import run_three_methods as experiments


OUT = experiments.ROOT / "output/class_text_token_fusion_0p5_layers6_11_seed123"
LAYERS = (6, 7, 9, 10, 11)
GPUS = (2, 3, 4, 5, 6)
CASES = [(layer, dataset, shots, seed) for dataset in experiments.DATASETS
         for shots in experiments.SHOTS for seed in experiments.SEEDS for layer in LAYERS]


def run_dir(case):
    layer, dataset, shots, seed = case
    return OUT / f"layer_{layer}" / dataset / f"shots_{shots}" / f"seed{seed}"


def summarize():
    rows = []
    for case in CASES:
        path = run_dir(case) / "test_metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if not rows:
        return
    fields = ("insert_layer", "dataset", "shots", "seed", "selected_epoch",
              "validation_accuracy", "test_accuracy")
    with (OUT / "results_detailed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for layer in range(6, 12):
        for dataset in experiments.DATASETS:
            for shots in experiments.SHOTS:
                if layer == 8:
                    source = experiments.ROOT / "output/class_text_token_fusion_0p5_seed123/fusion" / dataset / f"shots_{shots}"
                    values = [json.loads((source / f"seed{seed}/test_metrics.json").read_text())["test_accuracy"]
                              for seed in experiments.SEEDS]
                else:
                    values = [row["test_accuracy"] for row in rows if
                              (row["insert_layer"], row["dataset"], row["shots"]) == (layer, dataset, shots)]
                if len(values) == 3:
                    summary.append(dict(insert_layer=layer, dataset=dataset, shots=shots,
                                        mean=statistics.mean(values), std=statistics.stdev(values)))
    with (OUT / "results_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("insert_layer", "dataset", "shots", "mean", "std"))
        writer.writeheader()
        writer.writerows(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for case in CASES:
        pending.put(case)
    lock = threading.Lock()
    records = []

    def worker(gpu):
        while True:
            try:
                case = pending.get_nowait()
            except queue.Empty:
                return
            layer, dataset, shots, seed = case
            try:
                status = experiments.run_case(gpu, ("fusion", dataset, shots, seed),
                                              insert_layer=layer, output_dir=run_dir(case))
                record = dict(case=case, gpu=gpu, status=status)
            except Exception as exc:
                record = dict(case=case, gpu=gpu, status="failed", error=str(exc))
            with lock:
                records.append(record)
                experiments.write_json(OUT / "_manager/status.json", dict(
                    total=len(CASES), completed=sum(r["status"] != "failed" for r in records),
                    processed=len(records), records=records))
                summarize()
                print(json.dumps(record), flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
