import typing

import torch

from lib.MuxkitTools.audio_tools.bc_augmentation.bc_augmented_dataset import BCLearningDataset
from .PreprocessedDataset import create_preprocessed_acoustic_dataset


def make_bc_augmented_dataset(dataset, pre_augmentation_process: torch.nn.Module | typing.Callable,
                              post_augmentation_process: torch.nn.Module | typing.Callable,
                              sample_rate: int, n_class: int, device: str | torch.DeviceObjType):
    """
    Warp dataset with BCLearningDataset:
    1. preprocess dataset with pre_augmentation_process
    2. Wrap dataset with BCAugmentation
    3. post-preprocess augmented dataset with post_augmentation_process
    Args:
        dataset:
        pre_augmentation_process:
        post_augmentation_process:
        sample_rate:
        n_class:
        device:

    Returns:

    """
    pre_processed_dataset = create_preprocessed_acoustic_dataset(pre_augmentation_process, dataset,
                                                                 device, n_class)
    bc_set = BCLearningDataset(
        pre_processed_dataset,
        sample_rate,
        n_class,
        device
    )
    post_processed_dataset = create_preprocessed_acoustic_dataset(
        post_augmentation_process,
        bc_set,
        device,
        n_class
    )

    return post_processed_dataset
