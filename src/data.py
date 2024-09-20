import os
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader
from datasets import load_from_disk
from .utils import jsonlload, jsonldump, jsonl2df, load_dataset


def ImageCaptioningDataLoader(
    image_path: str,
    label_path: str,
    feature_extractor,
    tokenizer,
    batch_size: int,
    mode: str = "train",
    num_proc: int = 8,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    Build Data Loader with caching and support for different model types

    Args:
        image_path (str): Path to the directory containing images
        label_path (str): Path to the JSONL file containing labels
        feature_extractor (Optional[AutoFeatureExtractor]): Feature extractor for processing images
        tokenizer (Optional[AutoTokenizer]): Tokenizer for processing text
        batch_size (int): Number of samples per batch
        model_type (str): Type of the model (e.g., 'VisionEncoderDecoder', 'EXAONE', 'LLaVA', etc.)
        mode (str, optional): Mode of operation ('train' or 'valid'). Defaults to "train".
        num_proc (int, optional): Number of processes for data loading. Defaults to 8.
        shuffle (Optional[bool], optional): Whether to shuffle the data. Defaults to None.

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    Image.MAX_IMAGE_PIXELS = None
    
    # Define cache directory
    cache_dir = os.path.join(os.path.dirname(label_path), f"cached_{mode}_dataset")

    if os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        dataset = load_from_disk(cache_dir)
    else:
        print(f"Creating and caching dataset to {cache_dir}")

        def preprocess_function(examples):
            images = [Image.open(os.path.join(image_path, inp+".jpg")).convert("RGB").resize((224,224)) for inp in examples["input"]]
            pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
            
            if mode == "train":
                tokenizer_input = tokenizer([tokenizer.bos_token+s+tokenizer.eos_token for s in examples["output"]],
                                            padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_token_type_ids=False)
                return {
                    "pixel_values": pixel_values,
                    "decoder_input_ids": tokenizer_input["input_ids"],
                    "decoder_attention_mask": tokenizer_input["attention_mask"],
                }
            else:
                return {
                    "pixel_values": pixel_values,
                    "image_ids": examples["input"]
                }

        dataset = load_dataset(label_path, mode=mode)
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )
        
        dataset.save_to_disk(cache_dir)

    dataset.set_format("torch")
    if shuffle is None:
        shuffle = (mode == "train")
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_proc, pin_memory=True)

    return dataloader
