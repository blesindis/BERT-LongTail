import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import MixedData, BookCorpus, RestaurantForLM, Wikitext, Wikitext103
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)

LOAD_CHECKPOINT = False

# train and validation size for pretrain
TRAIN_LEN = 200000
VAL_LEN = 500

# folder paths
CENTER_MODEL_PATH = "outputs/0112-bert-wiki103(256)/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, 'centers-4-vanilla(use input).pth')
LOAD_FOLDER = "0112-mot-vanilla-wiki103(256)/checkpoint-10000"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CONFIG_PATH = 'config/bert_256.json'

# training parameters
num_epochs = 3
lr = 1.5e-4
weight_decay = 0.01


def main():
    set_seed(45)
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = Wikitext103(config=config)
    # dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    centers = load_layer_data(CENTER_PATH)
    model = base_models.BertWithMOE(config, centers)    
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    print(len(train_loader), len(val_loader))
    
    accelerator = Accelerator()
    
    # prepare model and data on device
    model, val_loader, train_loader = accelerator.prepare(model, val_loader, train_loader)

        
    with torch.no_grad():        
        loss_valid = validate(model, val_loader, accelerator)                
        print(loss_valid)
    

if __name__ == "__main__":
    main()
    
    