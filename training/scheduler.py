import torch
import torch.optim as optim
import math

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, lr_max, lr_min, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.lr_max * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        return [lr for _ in self.base_lrs]
