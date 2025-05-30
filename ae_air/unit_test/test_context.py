import unittest

import torch

from .. import hyperparameters as hyp
from ..Context import TrainContext


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = TrainContext()

    def test_dataloader_on_eval(self):
        b0, l0 = next(iter(self.ctx.test_loader))
        print("Output shape from test loader:", b0.shape)
        self.assertEqual(b0.shape, torch.Size([hyp.BatchSize, 1, 5 * 4, 75 * 2]))

    def test_dataloader_train(self):
        b0, l0 = next(iter(self.ctx.train_loader))
        print("Output shape from train loader before preprocessing:", b0.shape)
        b0 = self.ctx.data_preprocessor(b0.to(self.ctx.device))
        print("Output shape from train loader after preprocessing:", b0.shape)
        self.assertEqual(b0.shape, torch.Size([hyp.BatchSize, 1, 5 * 4, 75 * 2]))

    def test_forward_on_eval(self):
        b0, l0 = next(iter(self.ctx.test_loader))
        print("Output shape from test loader:", b0.shape)
        out = self.ctx.model(b0)
        self.assertEqual(out.shape, torch.Size([hyp.BatchSize, hyp.N_Classes]))

    def test_forward_on_train(self):
        b0, l0 = next(iter(self.ctx.train_loader))
        print("Output shape from train loader before preprocessing:", b0.shape)
        b0 = self.ctx.data_preprocessor(b0.to(self.ctx.device))
        print("Output shape from train loader after preprocessing:", b0.shape)
        out = self.ctx.model(b0)
        self.assertEqual(out.shape, torch.Size([hyp.BatchSize, hyp.N_Classes]))


if __name__ == '__main__':
    unittest.main()
