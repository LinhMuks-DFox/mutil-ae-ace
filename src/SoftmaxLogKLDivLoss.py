import torch.nn


class SoftmaxLogKLDivLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, predicted, target):
        log_pred = torch.nn.functional.log_softmax(predicted, dim=1)
        return self.kl_div(log_pred, target)