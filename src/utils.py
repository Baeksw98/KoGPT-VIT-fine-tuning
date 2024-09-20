import logging
import sys
import yaml
import pytorch_lightning as pl
import json
import pandas as pd
from datasets import Dataset
import os
import re
import glob

def get_logger(name: str) -> logging.Logger:
    """
    Return logger for logging

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    return logger

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def resolve_path(path, config_or_model_name):
    """
    Resolve variables in paths based on the config or model name.
    Supports ${model_name} and other variables defined in config.
    """
    if isinstance(config_or_model_name, str):
        # If it's a string, assume it's the model name
        return path.replace('${model_name}', config_or_model_name)
    elif isinstance(config_or_model_name, dict):
        # If it's a dictionary, replace all matching keys
        for key, value in config_or_model_name.items():
            if isinstance(value, str):
                path = path.replace(f'${{{key}}}', value)
        return path
    else:
        raise ValueError("config_or_model_name must be either a string or a dictionary")

def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]
    return j_list

def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

def jsonl2df(j_list, mode):
    data_dict = {"input": []}
    if mode=="train":
        data_dict["output"] = []
        for j in j_list:
            for caption in j["output"]:
                data_dict["input"].append(j["input"]["id"])
                data_dict["output"].append(caption)
    else:
        for j in j_list:
            data_dict["input"].append(j["input"]["id"])
    
    df = pd.DataFrame(data_dict)
    return df

def load_dataset(fname, mode):
    j_list = jsonlload(fname)
    df = jsonl2df(j_list, mode)
    dataset = Dataset.from_pandas(df)
    return dataset

def get_latest_checkpoint(output_dir):
    checkpoint_folders = glob.glob(os.path.join(output_dir, 'M-Epoch*'))
    if not checkpoint_folders:
        return None
    
    valid_checkpoints = []
    for folder in checkpoint_folders:
        match = re.search(r'M-Epoch(\d+)', os.path.basename(folder))
        if match:
            epoch_num = int(match.group(1))
            valid_checkpoints.append((epoch_num, folder))
    
    if not valid_checkpoints:
        return None
    
    # Sort by epoch number and get the latest
    latest_checkpoint = max(valid_checkpoints, key=lambda x: x[0])[1]
    
    # Check if pytorch_model.bin exists in the latest folder
    if os.path.exists(os.path.join(latest_checkpoint, "pytorch_model.bin")):
        return latest_checkpoint
    return None