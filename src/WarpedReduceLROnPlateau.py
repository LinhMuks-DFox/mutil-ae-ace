import torch.optim.lr_scheduler as lr_schedular


class WarpedReduceLROnPlateau(lr_schedular.ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose="deprecated", metrics_name: str | None = None):
        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps)

        self.metrics_name: str = metrics_name if metrics_name is not None else "validate_loss"
