import unittest

import torch

from .. import hyperparameters as hyp
from ..Context import TrainContext


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = TrainContext()

    def test_dataloader(self):
        b0, l0 = next(iter(self.ctx.test_loader))
        self.assertEqual(b0.shape, torch.Size([hyp.BatchSize, 1, 5 * 4, 75]))

        b0, l0 = next(iter(self.ctx.train_loader))
        b0 = self.ctx.data_preprocessor(b0.to(self.ctx.device))
        self.assertEqual(b0.shape, torch.Size([hyp.BatchSize, 1, 5 * 4, 75]))

    def test_forward(self):
        b0, l0 = next(iter(self.ctx.test_loader))
        out = self.ctx.model(b0)
        self.assertEqual(out.shape, torch.Size([hyp.BatchSize, hyp.N_Classes]))

        b0, l0 = next(iter(self.ctx.train_loader))
        b0 = self.ctx.data_preprocessor(b0.to(self.ctx.device))
        out = self.ctx.model(b0)
        self.assertEqual(out.shape, torch.Size([hyp.BatchSize, hyp.N_Classes]))


if __name__ == '__main__':
    unittest.main()
