import argparse
import os
import MeCab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_metric import PyRouge

from src.data import jsonlload
from src.utils import get_logger, load_config

def main(config):
    # Get configurations from config
    data_config = config['data_config']

    # Get command line arguments
    args = argparse.Namespace(**config['scoring_parameters'])
    
    # Initialize logger
    logger = get_logger("scoring")

    # Initialize Rouge and BLEU
    rouge = PyRouge(rouge_n=(1, 2, 4))

    logger.info(f'[+] Load Mecab from "{args.mecab_dict_path}"')
    tagger = MeCab.Tagger(f'-d {args.mecab_dict_path}')

    logger.info(f'[+] Load Dataset')
    base_dir = data_config['base_dir']
    jsonl_dir = os.path.join(base_dir, data_config['jsonl_dir'])
    
    reference_path = os.path.join(jsonl_dir, data_config['reference_file'])
    references_j_list = jsonlload(reference_path)
    references = [j["output"] for j in references_j_list]

    candidate_j_list = jsonlload(args.candidate_path)
    candidate = [j["output"] for j in candidate_j_list]

    logger.info(f'[+] Start POS Tagging')
    for idx, sentences in enumerate(references):
        output = []
        for s in sentences:
            tokenized = []
            for mor in tagger.parse(s.strip()).split("\n"):
                if "\t" in mor:
                    splitted = mor.split("\t")
                    token = splitted[0]
                    tokenized.append(token)
            output.append(tokenized)
        references[idx] = output

    for idx, s in enumerate(candidate):
        tokenized = []
        for mor in tagger.parse(s.strip()).split("\n"):
            if "\t" in mor:
                splitted = mor.split("\t")
                token = splitted[0]
                tokenized.append(token)
        candidate[idx] = tokenized

    smoother = SmoothingFunction()
    bleu_score = 0
    for idx, ref in enumerate(references):
        bleu_score += sentence_bleu(ref, candidate[idx], weights=(1.0, 0, 0, 0), smoothing_function=smoother.method1)
    logger.info(f'BLEU Score\t{bleu_score / len(references)}')

    rouge_score = rouge.evaluate(list(map(lambda cdt: " ".join(cdt), candidate)),
                                list(map(lambda refs: [" ".join(ref) for ref in refs], references)))
    logger.info(f'ROUGE Score\t{rouge_score["rouge-1"]["f"]}')

if __name__ == "__main__":
    config_path = "configs/config_score_base.yaml"
    config = load_config(config_path)
    main(config)