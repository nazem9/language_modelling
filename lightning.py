# lightning_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

# Import the LM model from model.py
from model import LM

class LanguageModelLightning(pl.LightningModule):
    """
    A PyTorch Lightning module wrapping the LM model, enabling easier
    training/validation loops, checkpointing, etc.
    """
    def __init__(self, vocab_size, lr=3e-4, steps=5000):
        super().__init__()
        self.save_hyperparameters()
        self.model = LM(vocab_size=vocab_size)
        self.lr = lr
        self.steps = steps

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }