import os
import unittest

import torch

from esc50.ESC50IO.ESC50 import ESC50, label_balanced_split_esc50
from src.PreprocessedDataset import create_cached_preprocessed_dataset
from .. import hyperparameters as hyp
from .. import options as opt
from ..DataPreprocessor import DataPreprocessor


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ESC50(opt.DataSetPath, download=(not os.path.exists(opt.DataSetPath)),
                             transform=None, subset="all")
        train_set, validate_set, test_set = label_balanced_split_esc50(self.dataset, [
            0.7, 0.15, 0.15])
        self.data_preprocessor = DataPreprocessor().to(opt.Device).to(torch.float32)
        self.train_set, self.validate_set, self.test_set = [
            create_cached_preprocessed_dataset(self.data_preprocessor, i_set, 2000, "cuda", 50) for i_set in [
                train_set, validate_set, test_set
            ]
        ]
        self.data0, self.label0 = self.train_set[0]

        print(self.data0.shape, self.label0.shape)

    def test_label_shape(self):
        self.assertEquals(
            torch.Size([1, hyp.AudioDuration * hyp.NewSampleRate]), self.data0.shape
        )
        self.assertEquals(
            torch.Size([50]), self.label0.shape
        )

    def test_data_device_and_dtype(self):
        self.assertEqual(
            self.data0.device, self.label0.device
        )
        self.assertEqual(self.data0.dtype, torch.float32)
        self.assertEqual(self.label0.dtype, torch.int64)
