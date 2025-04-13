import torch.nn
import torchvision

def make_classifier(model_type: str = None, class_cnt: int = None):
    _CASE = {
        "RES18": torchvision.models.resnet18,
        "RES34": torchvision.models.resnet34,
        "RES50": torchvision.models.resnet50,
    }
    _net_constructor = _CASE.get(model_type)
    _projection = torch.nn.Conv2d(kernel_size=(1, 1), in_channels=1, out_channels=3)
    return torch.nn.Sequential(_projection, _net_constructor(num_classes=class_cnt))