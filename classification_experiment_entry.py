#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import typing
import subprocess

import torch
import torch.amp

from latent_idea_classification_context.Context import TrainContext as latent_idea_classification_context

from src.ABCContext import Context
from src.Trainer import Trainer


def make_context(experiment_type: str) -> Context:
    return {
        "ltidl": latent_idea_classification_context,
    }[experiment_type]()


def get_hyperparameter_and_options_path(experiment_type: str) -> typing.Tuple[str, str]:
    return {
        "ltidl": ("./latent_idea_classification_context/hyperparameters.py",
                  "./latent_idea_classification_context/options.py"),
    }[experiment_type]


def config_logger(context: Context):
    log_file_path = os.path.join(context.dump_path, "train.log")
    logging.basicConfig(
        format='%(asctime)s - %(message)s:\n',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def make_trainer(context,
                 hyp_and_opt_path: typing.Tuple[str, str],
                 only_keep_best_epoch: bool = False,
                 best_model_elect_score: str = "f1"):
    return Trainer(ctx=context, logger=logging.getLogger(), best_model_elect_score=best_model_elect_score,
                   hyper_parameter_option_path=hyp_and_opt_path, only_keep_best_epoch=only_keep_best_epoch)


def make_dump_path(context):
    if not os.path.exists(context.dump_path):
        os.makedirs(context.dump_path)


def profiling_main():
    from torch.profiler import profile, ProfilerActivity
    logging.info("Run Train script in profiling-mode")
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(os.getcwd()))
    ):
        main()


def main():
    args = parse_arguments()
    experiment_type = {
        "ltidl": "ltidl"
    }[args.experiment.lower()]
    print("Context making...")
    context: Context = make_context(experiment_type)
    print("Dump path creating...")
    make_dump_path(context)
    print("Configuring logger...")
    config_logger(context)
    train_app = make_trainer(context, get_hyperparameter_and_options_path(experiment_type),
                             only_keep_best_epoch=args.only_keep_best_epoch)
    train_app.main()

    # 实验完成后发送通知邮件
    subprocess.run([
        "python", "email_sender.py",
        "--subject", "Experiment Notification ✅",
        "--body", f"Your experiment '{args.experiment}' has finished successfully!"
    ])


def parse_arguments():
    parser = argparse.ArgumentParser(description="TrainApp configuration")
    parser.add_argument("-E", "--experiment", type=str, required=True,
                        choices=["ltidl"],
                        help="Specify the experiment type.")
    parser.add_argument("-p", "--profiling", action="store_true", default=False, help="Run script with torch profile")
    parser.add_argument("-okbe", "--only-keep-best-epoch", action="store_true", default=False, help="只保留最好的epoch")
    return parser.parse_args()


if __name__ == "__main__":
    with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu",
            dtype=torch.float32):
        profiling_main() if parse_arguments().profiling else main()
