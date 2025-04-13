from collections import OrderedDict

import torch.nn as nn
from torchvision.models import resnet50, resnet18, resnet34
from . import hyperparameters as hyp


def make_model(resnet_type, n_cls):
    projection = nn.Conv2d(hyp.N_MIC, 3, kernel_size=1)
    resnet = {
        18: resnet18,
        34: resnet34,
        50: resnet50
    }[resnet_type](num_classes=n_cls)

    return nn.Sequential(OrderedDict([
        ("Projection", projection),
        ("Resnet", resnet)
    ]))
