
import argparse
import os
import torch
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer

from src.data import ImageCaptioningDataLoader
from src.module import ImageCaptioningModule
from src.utils import get_logger, load_config, resolve_path, get_latest_checkpoint
from src.model_utils import MidEpochGCCallback

# Set the matmul precision to high
torch.set_float32_matmul_precision('high')

# Ignore FutureWarning for huggingface_hub.file_download
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def main(config):
    # Get configurations from config
    model_config = config['model_config']
    data_config = config['data_config']
    model_name = config['model_name']
    
    # Get command line arguments
    args = argparse.Namespace(**config['common_parameters'])
    wandb_opts = argparse.Namespace(**config['wandb_options'])

    # Initialize logger
    logger = get_logger("train")
    
    # Resolve output directory
    args.output_dir = resolve_path(args.output_dir, model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguments ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f"[+] Load Feature Extractor")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_config['vision_encoder'])

    logger.info(f'[+] Load Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['text_decoder'],
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>'
    )

    logger.info(f'[+] Load Dataset')
    num_proc = min(args.num_proc, 32)
    logger.info(f"[+] Using {num_proc} processes for data loading")
        
    base_dir = data_config['base_dir']
    image_dir = os.path.join(base_dir, data_config['image_dir'])
    jsonl_dir = os.path.join(base_dir, data_config['jsonl_dir'])

    train_dataloader = ImageCaptioningDataLoader(
        image_dir,
        os.path.join(jsonl_dir, data_config['train_file']),
        feature_extractor,
        tokenizer,
        args.batch_size,
        num_proc=num_proc,
    )

    valid_dataloader = ImageCaptioningDataLoader(
        image_dir,
        os.path.join(jsonl_dir, data_config['val_file']),
        feature_extractor,
        tokenizer,
        args.valid_batch_size,
        num_proc=num_proc,
        shuffle=False  # Disable shuffling for validation
    )

    total_steps = len(train_dataloader) * args.epochs
    
    # Check for the latest checkpoint
    latest_checkpoint = get_latest_checkpoint(args.output_dir)
    
    if latest_checkpoint:
        logger.info(f"[+] Found latest checkpoint: {latest_checkpoint}")
        model = VisionEncoderDecoderModel.from_pretrained(latest_checkpoint)
        logger.info("[+] Resuming training from the latest checkpoint")
    else:
        logger.info(f'[+] Did not find any valid checkpoint, initializing model from "{model_config["vision_encoder"]}" and "{model_config["text_decoder"]}"')
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            model_config['vision_encoder'], model_config['text_decoder']
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info("[+] Starting training from the pretrained models")

    checkpoint_dir = os.path.join(args.output_dir, "model_ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='M-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        every_n_epochs=1,
        save_weights_only=True
    )
    
    logger.info(f"[+] Load Pytorch Lightning Module")
    lightning_module = ImageCaptioningModule(
        model=model,
        feature_extractor=feature_extractor,
        total_steps=total_steps,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_rate=args.warmup_rate,
        model_save_dir=args.output_dir,
        lr_scheduler_type=args.lr_scheduler_type,  
        cosine_annealing_WR=args.cosine_annealing_WR if hasattr(args, 'cosine_annealing_WR') else None,
    )

    # Get the last epoch number
    last_epoch = lightning_module.get_last_epoch()

    logger.info(f"[+] Start Training from epoch {last_epoch + 1}")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
    if wandb_opts.wandb_project:
        wandb_opts.wandb_save_dir = resolve_path(wandb_opts.wandb_save_dir, model_name)
        train_loggers.append(
            WandbLogger(
                name=wandb_opts.wandb_run_name or os.path.basename(args.output_dir),
                project=wandb_opts.wandb_project,
                entity=wandb_opts.wandb_entity,
                save_dir=wandb_opts.wandb_save_dir,
            )
        )
        
    # # Calculate the remaining epochs
    # remaining_epochs = args.epochs - last_epoch - 1  # -1 because last_epoch is 0-indexed
        
    logger.info(f"[+] Instantiating Pytorch Lightning Trainer")
    trainer = pl.Trainer(
        precision="16",  # Change to 16-bit precision
        logger=train_loggers,
        max_epochs=args.epochs, #remaining_epochs
        log_every_n_steps=args.logging_interval,  
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
            MidEpochGCCallback(),
        ],
        accelerator="gpu",
        devices=args.gpus,
        strategy='deepspeed' if args.gpus > 1 else None,
    )

    try:
        trainer.fit(lightning_module, train_dataloader, valid_dataloader)
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    
if __name__ == "__main__":
    config_path = "configs/train/config_train_VIT_GPT2_V1.yaml"
    config = load_config(config_path)
    main(config)