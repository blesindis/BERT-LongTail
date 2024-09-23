import os
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import Wikipedia ,BookCorpus, BERTPretrain
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
)

from collections import Counter

# Parameters
n_gram_size = 4
min_frequency = 4
max_frequency = 30

CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 3
lr = 1e-4
weight_decay = 0.01


def generate_ngrams(input_ids, n):
    return [tuple(input_ids[i:i+n]) for i in range(len(input_ids)-n+1)]



def main():
    set_seed(45)
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = BERTPretrain(config=config)
    
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    print(len(train_loader), len(val_loader))
    
    accelerator = Accelerator()
    
    # prepare model and data on device
    val_loader, train_loader = accelerator.prepare(val_loader, train_loader)


    # Initialize Counter
    ngram_counter = Counter()
    for batch in tqdm(train_loader, desc="Processing batches"):
        input_ids_batch = batch['input_ids']  # Extract input_ids from the batch
        for input_ids in input_ids_batch:
            n_grams = generate_ngrams(input_ids.tolist(), n_gram_size)  # Convert tensor to list and generate n-grams
            ngram_counter.update(n_grams)

    # Filter rare n-grams
    rare_ngrams = {ngram: count for ngram, count in ngram_counter.items() if count > max_frequency and count < 200}

    rare_ngrams = dict(list(rare_ngrams.items())[:10000])

    # Display total number of rare n-grams found
    print(f"Found {len(rare_ngrams)} rare n-grams")

    # Save rare n-grams to a file
    rare_ngrams_path = "rare_ngrams_10000_n4_freq30-200.json"

    # Option 1: Save as human-readable JSON (useful for inspection)
    with open(rare_ngrams_path, 'w') as f:
        json.dump({str(k): v for k, v in rare_ngrams.items()}, f, indent=4)
    

if __name__ == "__main__":
    main()
    
    