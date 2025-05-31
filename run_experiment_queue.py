#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path

ENTRY = "classification_experiment_entry.py"
AVAILABLE_EXPERIMENTS = ["ltidl", "speidl", "ltari", "sdp", "e2e", "rpi"]

SAVE_SUFFIX = {
    "ltidl": "ideal-latent",
    "speidl": "spectro-ideal",
    "ltari": "air-propagate-latent",
    "sdp": "sound-power",
    "e2e": "End-2-end",
    "rpi": "rpi-ae",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a queue of experiments sequentially")
    parser.add_argument("--queue", nargs='+', choices=AVAILABLE_EXPERIMENTS, required=True,
                        help="Experiments to run in sequence")
    parser.add_argument("-okbe", "--only-keep-best-epoch", action="store_true",
                        help="Pass -okbe to classification_experiment_entry")
    parser.add_argument("-pf", "--profiling", action="store_true",
                        help="Run with profiling enabled")
    parser.add_argument("--nohup", action="store_true",
                        help="Use nohup when launching each experiment")
    parser.add_argument("--google-drive-path", type=str, default=None,
                        help="Copy experiment results to this path after each run")
    return parser.parse_args()


def run_command(cmd, log_file=None):
    """Run command and wait for it to finish."""
    if log_file:
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
            process.wait()
            return process.returncode
    else:
        return subprocess.call(cmd)


def main():
    args = parse_args()
    for exp in args.queue:
        start_time = datetime.now()
        save_dir = f"./trained/{start_time.strftime('%Y-%m-%d-%H-%M-%S')}-{SAVE_SUFFIX[exp]}"
        cmd = ["python3", "-u", ENTRY, "-E", exp]
        if args.profiling:
            cmd.append("--profiling")
        if args.only_keep_best_epoch:
            cmd.append("-okbe")

        if args.nohup:
            log = f"nohup_{exp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            cmd = ["nohup"] + cmd
            print(f"Running {' '.join(cmd)} -> {log}")
            ret = run_command(cmd, log_file=log)
        else:
            print(f"Running {' '.join(cmd)}")
            ret = run_command(cmd)

        if ret != 0:
            print(f"Experiment {exp} exited with code {ret}")
            sys.exit(ret)
        if args.google_drive_path:
            # match directory using minute-level timestamp to avoid second level mismatch
            prefix = start_time.strftime('%Y-%m-%d-%H-%M')
            suffix = SAVE_SUFFIX[exp]
            trained_root = Path('./trained')
            candidates = sorted(
                [d for d in trained_root.glob(f"{prefix}-*-{suffix}") if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            actual_dir = candidates[0] if candidates else Path(save_dir)
            dest = Path(args.google_drive_path) / actual_dir.name
            print(f"Copying {actual_dir} -> {dest}")
            try:
                shutil.copytree(actual_dir, dest, dirs_exist_ok=True)
            except Exception as e:
                print(f"Failed to copy results: {e}")

    print("All experiments finished")


if __name__ == "__main__":
    main()
