#!/usr/bin/env python3
import argparse
import subprocess
import sys
from datetime import datetime

ENTRY = "classification_experiment_entry.py"
AVAILABLE_EXPERIMENTS = ["ltidl", "speidl", "ltari", "sdp", "e2e", "rpi"]


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
        cmd = ["python", "-u", ENTRY, "-E", exp]
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

    print("All experiments finished")


if __name__ == "__main__":
    main()
