import typing

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

from MuxkitTools.dataset_tools.CachableDataset import CacheableDataset

# A Preprocessor object or function shall take a tensor as input, mapping those tensor into processed tensor
PreprocessorType = typing.Callable[[torch.Tensor], torch.Tensor]


class PreprocessedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_preprocessor: PreprocessorType | None,
                 data_loader: torch.utils.data.Dataset,
                 device: str = "cpu",
                 n_class: int = 50,
                 to_one_hot: bool = True) -> None:
        self.preprocessor = data_preprocessor
        self.data_loader = data_loader
        self.device = device
        self.n_class = n_class
        self.to_one_hot = to_one_hot

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, index):
        sample, label = self.data_loader[index]
        if isinstance(label, int) and self.to_one_hot:
            label = nn.functional.one_hot(torch.tensor(label), num_classes=self.n_class)
        sample, label = sample.to(self.device), label.to(self.device).float()
        if self.preprocessor is not None:
            sample = self.preprocessor(sample)
        return sample, label


def create_preprocessed_acoustic_dataset(
        data_preprocessor: PreprocessorType | None,
        data_loader: torch.utils.data.Dataset,
        device: str,
        n_class: int,
        to_onehot: bool = True
):
    return PreprocessedDataset(data_preprocessor, data_loader, device=device, n_class=n_class, to_one_hot=to_onehot)


def create_cached_preprocessed_dataset(data_preprocessor: PreprocessorType,
                                       data_loader: torch.utils.data.Dataset,
                                       cache_length: int,
                                       device: str,
                                       n_class: int,
                                       to_onehot: bool = True):
    dataset = CacheableDataset(
        PreprocessedDataset(data_preprocessor, data_loader, device=device, n_class=n_class, to_one_hot=to_onehot),
        multiprocessing=False,
        max_cache_size=cache_length,
        device=device,
    )
    # dataset = _BlinkyAcousticDataset(data_preprocessor, data_loader, device=device, n_class=n_class)
    return dataset
