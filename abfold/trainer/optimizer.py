from torch.optim import AdamW, Adam

class OptipizerInverseSquarRootDecay(AdamW):
    def __init__(self, parameters, base_lr, warmup_steps=0, flat_steps=0, decay_type='poly'):
        super().__init__(parameters, base_lr)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.flat_steps = flat_steps
        
        self.cur_step = 0

        assert decay_type in ['poly', 'half']

        self.decay_type = decay_type

        self.start_lr = 1e-5

    def get_values(self):
        if self.cur_step < self.warmup_steps:
            lr = self.start_lr + (self.base_lr - self.start_lr) / self.warmup_steps * self.cur_step
        elif self.cur_step < self.warmup_steps + self.flat_steps:
            lr = self.base_lr
        else:
            if self.decay_type == 'poly':
                lr = self.base_lr * (1 + self.cur_step - self.warmup_steps - self.flat_steps) ** -0.5
            elif self.decay_type == 'half':
                lr = self.base_lr / 2.
            else:
                raise NotImplemented(f'decay type {self.decay_type} not implemented')

        return lr

    def step(self):
        lr = self.get_values()

        for param_group in self.param_groups:
            param_group['lr'] = lr
        
        super().step()

        self.cur_step += 1
