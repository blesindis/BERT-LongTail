import os
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from transformer.MoMoTModelRouterMTLLastTwo import BertWithMoMoTModelRouterMTLLastTwo
from transformer.BERT import BertForMLM

# Local imports
from Dataset import BERTPretrain, LePuWiki, Wikipedia, WikiSST
from utils.sample_utils import *
from utils.train_utils import (
    set_seed,
    load_layer_data_last,
)

from collections import Counter


CONFIG_PATH = 'config/mtl.json'

# Assuming 'lm_datasets_train' is loaded and contains tokenized data
def calculate_word_frequency(dataset, tokenizer):
    word_freq = Counter()

    # Iterate through the dataset
    for batch in dataset:
        # Each 'input_ids' in the batch represents tokenized text
        input_ids = batch['input_ids']

        # Decode tokens to words
        words = [tokenizer.decode(token_ids) for token_ids in input_ids]

        # Update word frequency count
        for word in words:
            word_freq.update(word.split())

    return word_freq


def main():
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = Wikipedia(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    # val_loader_l, val_loader_p, val_loader_w = dataset.val_loader_legal, dataset.val_loader_pub, dataset.val_loader_wiki
    
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    # val_loader_l, val_loader_p, val_loader_w = accelerator.prepare(val_loader_l, val_loader_p, val_loader_w)
        
    # Assuming 'tokenizer' and 'lm_datasets_train' are already defined and loaded
    word_frequencies = calculate_word_frequency(dataset.train_set, dataset.tokenizer)

    # Print the most common words
    print(word_frequencies.most_common(10))
    
    # Convert Counter object to a dictionary (optional, depends on use case)
    word_freq_dict = dict(word_frequencies)

    # Save the word frequencies to a JSON file
    with open('word_frequencies.json', 'w') as f:
        json.dump(word_freq_dict, f, indent=4)

    # If you prefer to save it as a CSV:
    import csv

    with open('word_frequencies.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Frequency'])  # Writing header
        for word, frequency in word_frequencies.items():
            writer.writerow([word, frequency])

    print("Word frequencies saved to 'word_frequencies.json' and 'word_frequencies.csv'.")


if __name__ == "__main__":
    main()