import pytorch_lightning as pl
import os 
            
class MidEpochGCCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == len(trainer.train_dataloader) // 2:
            pl_module.on_train_epoch_mid()