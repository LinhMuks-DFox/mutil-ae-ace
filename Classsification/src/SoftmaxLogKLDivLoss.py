import torch.nn


class SoftmaxLogKLDivLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, predicted, target):
        predicted = torch.nn.functional.softmax(predicted, dim=1).log()
        return self.kl_div(predicted, target)
