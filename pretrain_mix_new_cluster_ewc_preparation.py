import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import BertConfig

# Local imports
import base_models
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    load_layer_data,
)
from Dataset import MixedData

SAMPLE_BATCHES = 100


# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1102-mixed-stage1"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CENTER_FILE = 'centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
REPLAY_FILE = 'replay_dynamic_100.pth'
REPLAY_PATH = os.path.join(LOAD_PATH, REPLAY_FILE)
CONFIG_PATH = 'config/bert.json'

# training config
lr = 1e-3
weight_decay = 0


def main():
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = MixedData(config, TRAIN_LEN, VAL_LEN)
    centers = load_layer_data(CENTER_PATH)
    
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    
    accelerator = Accelerator()
    
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    
    
    model.eval()
    fim = {}
    for name, param in model.named_parameters():
        fim[name] = torch.zeros_like(param)
        
    for i, batch in enumerate(train_loader):
        if i >= SAMPLE_BATCHES:
            break
        
        model.zero_grad()
        loss, _ = model(**batch)
        loss.backward()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                fim[name] += param.grad ** 2
    
    for name in fim.keys():
        fim[name] /= SAMPLE_BATCHES
        
    torch.save(fim, os.path.join(LOAD_PATH, 'fim.pth'))
    
if __name__ == '__main__':
    main()