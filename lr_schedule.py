import math

class CosineAnnealingLRWithWarmup:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # warmup phase
            eta = 1.0 / self.warmup_epochs * (epoch + 1)
        else:
            # cosine annealing with restarts
            t = epoch - self.warmup_epochs
            T = self.max_epochs - self.warmup_epochs
            eta = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + math.cos(math.pi * t / T))
            if t % T == 0:
                self.optimizer.param_groups[0]['lr'] = eta

        self.optimizer.param_groups[0]['lr'] = eta
