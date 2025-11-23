import subprocess
import itertools
import os
import sys

datasets = ["cifar-100", "cifar-10", "mnist"]
models = ["MLP", "resnet18","resnet50","resnet14d","resnet26d"]
optimizers = ["sgd", "adam", "adamw", "muon", "sam"]
schedulers = ["steplr", "reducelronplateau"]
augmentations_options = [
    [],
    ["crop", "flip"],
    ["crop", "flip", "colorjitter"],
]

epochs = 150
batch_size = 128
lr = 0.01
weight_decay = 1e-4
momentum = 0.9
allow_cpu = False
save_dir = "models"

os.makedirs(save_dir, exist_ok=True)

def build_command(dataset, model, optimizer, scheduler, augmentations, run_id):
    cmd = [
        sys.executable, "train_pipeline.py",
        "--dataset", dataset,
        "--model", model,
        "--optimizer", optimizer,
        "--scheduler", scheduler,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--momentum", str(momentum),
        "--save-path", os.path.join(save_dir, f"{dataset}_{model}_{optimizer}_{scheduler}_{run_id}.pth")
    ]
    if allow_cpu:
        cmd.append("--allow-cpu")
    if augmentations:
        cmd += ["--augmentations"] + augmentations
    return cmd

if __name__ == "__main__":
    all_combinations = list(itertools.product(datasets, models, optimizers, schedulers, augmentations_options))
    total_runs = len(all_combinations)

    for run_id, (dataset, model, optimizer, scheduler, augmentations) in enumerate(all_combinations):
        print(f"\nRunning combination {run_id+1}/{total_runs}: "
              f"{dataset}, {model}, {optimizer}, {scheduler}, aug={augmentations}")
        cmd = build_command(dataset, model, optimizer, scheduler, augmentations, run_id)
        subprocess.run(cmd, check=True)
