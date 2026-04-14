class AWP:
    """
    Adversarial Weight Perturbation — perturbs model weights slightly
    in the direction that increases loss, forcing the model to be robust.
    lambda_=0.2 as per the Kaggle solution.
    """
    def __init__(self, model, optimizer, criterion, lambda_=0.2, start_epoch=15):
        self.model       = model
        self.optimizer   = optimizer
        self.criterion   = criterion
        self.lambda_     = lambda_
        self.start_epoch = start_epoch
        self._backup     = {}

    def perturb(self, seqs, masks, lbls):
        # Save original weights
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self._backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm != 0:
                    param.data.add_(
                        self.lambda_ * param.grad / norm
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()

    def step(self, seqs, masks, lbls, epoch):
        if epoch < self.start_epoch:
            return
        self.perturb(seqs, masks, lbls)
        loss = self.criterion(self.model(seqs, padding_mask=masks), lbls)
        self.optimizer.zero_grad()
        loss.backward()
        self.restore()