import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class DisturbanceLitModule(pl.LightningModule):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, learning_rate: float = 1e-3, weight_decay: float = 1e-4, linear_probe: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        # Simple early-warning prediction head taking previous embedding, change vector, and weather
        input_dim = (embedding_dim * 2) + 12 # assuming 12-dim monthly weather aggregates
        
        if linear_probe:
            self.head = nn.Linear(input_dim, 1)
            # Freeze layers if base model is fine-tuned later
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1)
            )

        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

    def forward(self, embedding_prev, embedding_change, weather):
        # Concatenate features
        x = torch.cat([embedding_prev, embedding_change, weather], dim=-1)
        logits = self.head(x).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["embedding_prev"], batch["embedding_change"], batch["weather"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
        
        preds = torch.sigmoid(logits)
        self.train_auc.update(preds, batch["label"].int())
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["embedding_prev"], batch["embedding_change"], batch["weather"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
        
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, batch["label"].int())
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer
