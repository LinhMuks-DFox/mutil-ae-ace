import unittest

import torch
import torchinfo
from ..Context import TrainContext


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.ctx = TrainContext()

    def test_label_shape(self):
        batch0 = next(iter(self.ctx.train_loader))
        print(
            batch0[0].shape, batch0[1].shape
        )

    def test_auto_encoder(self):
        torchinfo.summary(self.ctx.auto_encoder, torch.rand(1, 1, 80, 501).to(self.ctx.device))

    def test_data_device_and_dtype(self):
        pass
