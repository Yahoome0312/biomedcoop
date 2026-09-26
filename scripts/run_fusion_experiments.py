"""Run the 0.5 Text Deep Prompt + 0.5 Class Text Token comparison."""

from scripts import run_three_methods as experiments


experiments.OUT = experiments.ROOT / "output/class_text_token_fusion_0p5_seed123"
experiments.METHODS = ("fusion",)
experiments.CASES = [("fusion", dataset, shots, seed)
                     for dataset in experiments.DATASETS
                     for shots in experiments.SHOTS
                     for seed in experiments.SEEDS]
experiments.GPUS = (0, 1, 3, 4, 5)


if __name__ == "__main__":
    experiments.main()
