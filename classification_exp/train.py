#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import typing

import torch
import torch.amp

from acdnet.Context import TrainContext as acdnet_context
from embedding_blinkies_acoustic.Context import TrainContext as blinkies_acoustic_context
from framework_debug.Context import TrainContext as dbg_context
from no_room_blinky.Context import TrainContext as noom_blk_contenxt
from resnet_acoustic.Context import TrainContext as resnet_context
from soundpower_blinkies_acoustic.Context import TrainContext as soundpower_blinkies_acoustic_context
from spectrogram_acoustic.Context import TrainContext as spectrogram_acoustic_context
from src import Context
from src import Trainer


def make_context(experiment_type: str) -> Context:
    return {
        "blinkies": blinkies_acoustic_context,
        "sound_power": soundpower_blinkies_acoustic_context,
        "spectrogram": spectrogram_acoustic_context,
        "acdnet": acdnet_context,
        "resnet": resnet_context,
        "dbg": dbg_context,
        "no_room_blinky": noom_blk_contenxt
    }[experiment_type]()


def get_hyperparameter_and_options_path(experiment_type: str) -> typing.Tuple[str, str]:
    return {
        "blinkies": ("./embedding_blinkies_acoustic/hyperparameters.py",
                     "./embedding_blinkies_acoustic/options.py"),
        "sound_power": ("./soundpower_blinkies_acoustic/hyperparameters.py",
                        "./soundpower_blinkies_acoustic/options.py"),
        "spectrogram": ("./spectrogram_acoustic/hyperparameters.py",
                        "./spectrogram_acoustic/options.py"),
        "acdnet": ("./acdnet/hyperparameters.py",
                   "./acdnet/options.py"),
        "resnet": ("./resnet_acoustic/hyperparameters.py",
                   "./resnet_acoustic/options.py"),
        "dbg": ("./framework_debug/hyperparameter.py", "./framework_debug/options.py"),
        "no_room_blinky": ("./no_room_blinky/hyperparameters.py", "./no_room_blinky/options.py")
    }[experiment_type]


def config_logger(context: Context):
    logging.basicConfig(
        format='%(asctime)s: \n%(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(context.dump_path, "log.log"), mode="w"),
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
    print("Run Train scrip in profiling-mode")
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
        "acd": "acdnet",
        "blk": "blinkies",
        "sp": "sound_power",
        "spcg": "spectrogram",
        "rsnt": "resnet",
        "dbg": "dbg",
        "nrblk": "no_room_blinky"
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="TrainApp configuration")
    parser.add_argument("-E", "--experiment", type=str, required=True,
                        choices=["blk", "sp", "spcg", "acd", "rsnt", "dbg", "nrblk"],
                        help="Specify the experiment type.")
    parser.add_argument("-p", "--profiling", action="store_true", default=False, help="Run script with torch profile")
    parser.add_argument("-okbe", "--only-keep-best-epoch", action="store_true", default=False, help="只保留最好的epoch")
    return parser.parse_args()


if __name__ == "__main__":
    with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu",
            dtype=torch.float32):
        profiling_main() if parse_arguments().profiling else main()
