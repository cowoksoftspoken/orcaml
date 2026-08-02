import math
from .optimizer import _validate_non_negative_float, _validate_positive_int

class LRScheduler:
    """
    Base class for learning rate schedulers.
    """
    def __init__(self, optimizer):
        if not hasattr(optimizer, "lr"):
            raise TypeError("optimizer must expose an lr attribute")
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr

    def get_lr(self):
        raise NotImplementedError

class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.step_size = _validate_positive_int("step_size", step_size)
        self.gamma = _validate_non_negative_float("gamma", gamma)
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))

class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.T_max = _validate_positive_int("T_max", T_max)
        self.eta_min = _validate_non_negative_float("eta_min", eta_min)
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0

class LinearWarmup(LRScheduler):
    def __init__(self, optimizer, warmup_epochs):
        self.warmup_epochs = _validate_positive_int("warmup_epochs", warmup_epochs)
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            return self.base_lr
        if self.last_epoch == 0:
            return 0.0
        return self.base_lr * (self.last_epoch / self.warmup_epochs)
