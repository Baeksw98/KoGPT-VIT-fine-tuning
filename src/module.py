import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor
import psutil
import gc
import glob
import re

class ImageCaptioningModule(pl.LightningModule):
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        feature_extractor: AutoFeatureExtractor,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float,
        model_save_dir: str,
        lr_scheduler_type: str, 
        cosine_annealing_WR: dict = None,  
    ):
        super().__init__()

        self.model = model
        self.feature_extractor = feature_extractor
        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir
        self.lr_scheduler_type = lr_scheduler_type
        self.cosine_annealing_WR = cosine_annealing_WR  

        self.save_hyperparameters(
            {
                **model.config.to_dict(),
                "total_steps": total_steps,
                "max_learning_rate": self.max_learning_rate,
                "min_learning_rate": self.min_learning_rate,
                "warmup_rate": self.warmup_rate,
                "lr_scheduler_type": self.lr_scheduler_type,
                "cosine_annealing_WR": self.cosine_annealing_WR,  
            }
        )
        
        self.best_val_loss = float('inf')
        self.last_epoch = self.get_last_epoch()

    def get_last_epoch(self):
        checkpoint_folders = glob.glob(os.path.join(self.model_save_dir, 'M-Epoch*'))
        if not checkpoint_folders:
            return -1
        
        valid_epochs = []
        for folder in checkpoint_folders:
            match = re.search(r'M-Epoch(\d+)', os.path.basename(folder))
            if match:
                epoch_num = int(match.group(1))
                valid_epochs.append(epoch_num)
        
        return max(valid_epochs) if valid_epochs else -1

    def on_fit_start(self):
        # Set the epoch to start from last_epoch + 1
        self.trainer.fit_loop.epoch_progress.current.completed = self.last_epoch
        
    def training_step(self, batch, batch_idx):
        output = self.model(
            pixel_values=batch["pixel_values"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        process = psutil.Process()
        memory_info = process.memory_info()
        self.log('memory_usage', memory_info.rss / 1024**3, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    def on_train_epoch_end(self):
        # Perform garbage collection at the end of each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_mid(self):
        # This method is called halfway through the epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        output = self.model(
            pixel_values=batch["pixel_values"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log("val_loss_epoch", val_loss_mean, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc_epoch", val_acc_mean, prog_bar=True, logger=True, sync_dist=True)

        # Check if we're in the main training loop (not in sanity check)
        is_sanity_check = self.trainer.sanity_checking
        is_main_training = self.trainer.is_global_zero and not is_sanity_check
        
        if is_main_training:
            current_epoch = self.last_epoch + self.trainer.current_epoch + 1
            
            # Save the model in the HuggingFace format
            save_path = os.path.join(
                self.model_save_dir,
                f"M-Epoch{current_epoch}-loss={val_loss_mean:.2f}"
            )
            self.model.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

            if val_loss_mean < self.best_val_loss:
                self.best_val_loss = val_loss_mean
                best_model_path = os.path.join(self.model_save_dir, "best_model")
                self.model.save_pretrained(best_model_path)
                print(f"New best model saved to {best_model_path} with validation loss: {val_loss_mean:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def on_save_checkpoint(self, checkpoint):
        # Save only the model state dict
        checkpoint["model_state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load the model state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=self.max_learning_rate,
            weight_decay=0.01
        )
        
        if self.lr_scheduler_type == "cosine_annealing_WR":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cosine_annealing_WR['T_0'] * self.total_steps, 
                T_mult=self.cosine_annealing_WR['T_mult'],
                eta_min=self.min_learning_rate
            )
        elif self.lr_scheduler_type == "cyclic":
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.min_learning_rate,
                max_lr=self.max_learning_rate,
                step_size_up=int(self.total_steps * self.warmup_rate),
                step_size_down=self.total_steps - int(self.total_steps * self.warmup_rate),
                mode='triangular',
                cycle_momentum=False
            )
        elif self.lr_scheduler_type == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps,
                eta_min=self.min_learning_rate
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.lr_scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "Learning Rate"},
        }