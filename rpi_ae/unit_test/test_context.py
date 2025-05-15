import unittest

import torch

from .. import hyperparameters as hyp
from ..Context import TrainContext


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = TrainContext()
        print(self.ctx)

    def test_dataloader(self):
        b0, l0 = next(iter(self.ctx.test_loader))
        print(b0.shape)
        print(l0.shape)


        b0, l0 = next(iter(self.ctx.train_loader))
        print(b0.shape)
        print(l0.shape)
        # self.assertEqual(
        #     b0.shape, torch.Size([hyp.BatchSize, 1, 1024, 216])
        # )

    def test_forward(self):
        trb0, trl0 = next(iter(self.ctx.train_loader))
        out = self.ctx.model(trb0)
        print(out.shape)


        b0, l0 = next(iter(self.ctx.train_loader))
        out = self.ctx.model(b0)
        print(out.shape)

if __name__ == '__main__':
    unittest.main()
