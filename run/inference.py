import argparse
from tqdm import tqdm
import os
import torch
from transformers import AutoFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer
import warnings 

from src.data import ImageCaptioningDataLoader, jsonlload, jsonldump
from src.utils import get_logger, load_config, resolve_path

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def main(config):
    # Get configurations from config
    model_config = config['model_config']
    data_config = config['data_config']

    # Get command line arguments
    args = argparse.Namespace(**config['inference_parameters'])
    
    # Initialize logger
    logger = get_logger("inference")

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f"[+] Load Feature Extractor")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_config['vision_encoder'])

    logger.info(f'[+] Load Tokenizer')
    if args.tokenizer:
        logger.info(f'Using custom tokenizer: {args.tokenizer}')
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        logger.info(f'Using default tokenizer: {model_config["text_decoder"]}')
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            model_config['text_decoder'],
            bos_token='</s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )

    logger.info(f'[+] Load Dataset')
    
    num_proc = min(args.num_proc, 64)
    logger.info(f"[+] Using {num_proc} processes for data loading")
    
    base_dir = data_config['base_dir']
    image_dir = os.path.join(base_dir, data_config['image_dir'])
    jsonl_dir = os.path.join(base_dir, data_config['jsonl_dir'])

    dataloader = ImageCaptioningDataLoader(
        image_dir,
        os.path.join(jsonl_dir, data_config['test_file']),
        feature_extractor,
        decoder_tokenizer,
        args.batch_size,
        mode="test",
        num_proc=num_proc,
    )
    
    # Resolve the model checkpoint path
    model_ckpt_path = resolve_path(args.model_ckpt_path, config)
    logger.info(f'[+] Load Model from "{model_ckpt_path}"')
    
    if not os.path.exists(model_ckpt_path):
        logger.error(f"Model checkpoint not found at {model_ckpt_path}")
        return

    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_ckpt_path)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    logger.info("[+] Start Inference")
    total_summary_tokens = []
    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        summary_tokens = model.generate(
            pixel_values=pixel_values,
            max_length=args.output_max_seq_len,
            pad_token_id=decoder_tokenizer.pad_token_id,
            bos_token_id=decoder_tokenizer.bos_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
            num_beams=args.num_beams,
            use_cache=True,
        )
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())

    logger.info("[+] Start Decoding")
    decoded = [decoder_tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]
    
    j_list = jsonlload(os.path.join(jsonl_dir, data_config['test_file']))
    for idx, oup in enumerate(decoded):
        j_list[idx]["output"] = oup

    output_path = resolve_path(args.output_path, config)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jsonldump(j_list, output_path)
    logger.info(f"[+] Saved inference results to {output_path}")

if __name__ == "__main__":
    config_path = "configs/inference/config_inference_base.yaml"
    config = load_config(config_path)
    main(config)