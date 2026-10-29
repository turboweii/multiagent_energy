#!/usr/bin/env python3
"""Replay archived posted-offer models or reproduce their seed-zero training.

The release bundle places a copy of this script at its root as reproduce.py.
Use the experiment's Python environment; no dependencies are installed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(source, script, *arguments):
    command = [sys.executable, str(source / "scripts" / script), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=source, check=True)


def mirror_models(saved, output):
    initial = saved / "initial-users"
    for variant in ("full-history", "no-history"):
        src, dst = saved / variant, output / variant
        dst.mkdir(parents=True, exist_ok=True)
        for checkpoint in ("best_fixed", "best_joint"):
            if (src / checkpoint).is_dir():
                shutil.copytree(src / checkpoint, dst / checkpoint, dirs_exist_ok=True)
        for name in ("validation.csv", "platform_training.csv", "joint_validation.csv",
                     "joint_training.csv"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        metadata = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        # Only this new replay copy is rebased; saved models and provenance
        # remain byte-for-byte unchanged in the archived experiment.
        metadata["archived_users_path"] = metadata["users_path"]
        metadata["users_path"] = str(initial)
        (dst / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                           encoding="utf-8")


def evaluate(source, legacy, initial, output):
    common = ["--stage", "test", "--legacy", legacy, "--seed", 0]
    run(source, "evaluate_posted_offers.py", *common, "--users", initial,
        "--output", output / "baselines-initial", "--model-dir", output / "full-history",
        "--extra-model", f"no_history_ppo={output / 'no-history'}")
    if (output / "baselines-joint" / "selection.json").exists():
        run(source, "evaluate_posted_offers.py", *common, "--users",
            output / "full-history" / "best_joint", "--output", output / "baselines-joint",
            "--model-dir", output / "full-history", "--checkpoint", "best_joint")
    run(source, "summarize_posted_offers.py", "--experiment", output)
    if (output / "baselines-joint" / "test_outcomes.csv").exists():
        run(source, "plot_posted_tradeoff.py", "--experiment", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("replay", "train"), default="replay")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--joint-cycles", type=int, default=1,
                        help="Joint-training cycles for a new training run; this archived run used four")
    args = parser.parse_args()
    base = args.artifact_root.resolve()
    source, saved = base / "source", base / "seed0"
    legacy = base / "inputs" / "legacy-seed0"
    output = (args.output or base / ("replayed" if args.mode == "replay" else "retrained") / "seed0").resolve()
    if not source.is_dir() or not legacy.is_dir():
        parser.error("Use the release bundle root, which contains source/, seed0/ and inputs/.")
    if output == saved or saved in output.parents:
        parser.error("Choose a new output directory outside the saved seed0 experiment.")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "replay":
        mirror_models(saved, output)
        for cohort in ("baselines-initial", "baselines-joint"):
            src, dst = saved / cohort, output / cohort
            if not (src / "selection.json").exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("selection.json", "selection_manifest.json", "logit_prior.json", "logit_online_map.json"):
                shutil.copy2(src / name, dst / name)
        evaluate(source, legacy, saved / "initial-users", output)
        return
    initial = output / "initial-users"
    common = ["--legacy", legacy, "--seed", 0]
    run(source, "train_posted_offers.py", "--stage", "users", *common,
        "--users", base / "inputs" / "varied-initial-users", "--output", initial, "--steps", 8192)
    for variant, extra in (("full-history", ["--joint-cycles", args.joint_cycles]),
                           ("no-history", ["--joint-cycles", 0, "--no-history"])):
        run(source, "train_posted_offers.py", "--stage", "train", *common, "--users", initial,
            "--output", output / variant, "--steps", 131072, *extra)
    for cohort, users in (("baselines-initial", initial),
                          ("baselines-joint", output / "full-history" / "best_joint")):
        run(source, "evaluate_posted_offers.py", "--stage", "select", *common, "--users", users,
            "--output", output / cohort, "--workers", args.workers)
    evaluate(source, legacy, initial, output)


if __name__ == "__main__":
    main()
