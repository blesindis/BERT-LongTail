import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import BertConfig

# Local imports
import base_models
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    load_layer_data,
)
from Dataset import MixedData, ACLForLM

# cluster config
SAMPLE_BATCHES = 100
NUM_CLUSTERS = 48
DATA_PER_CLUSTER = 8


# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1027-mixed-warmup"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = '1102-mixed-stage1'
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CENTER_FILE = 'centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
REPLAY_FILE = 'replay_dynamic_100.pth'
REPLAY_PATH = os.path.join(STORE_PATH, REPLAY_FILE)
REPLAY_BL_FILE = 'replay_baseline-48*8.pth'
REPLAY_BL_PATH = os.path.join(STORE_PATH, REPLAY_BL_FILE)
CONFIG_PATH = 'config/bert.json'

# training config
lr = 1e-3
weight_decay = 0 


def main():  
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = MixedData(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    new_dataset = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    centers = load_layer_data(CENTER_PATH)
    
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    new_train_loader = new_dataset.train_loader
    
    accelerator = Accelerator()
    
    model, train_loader, val_loader, new_train_loader = accelerator.prepare(model, train_loader, val_loader, new_train_loader)

    # path_dict = torch.load(REPLAY_PATH)
    # print(len(path_dict))
    
    last_layer_outputs = []
    input_ids = []
    labels = []
    attention_mask = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            
            output = model.bert(batch['input_ids'], batch['attention_mask'])
            last_layer_outputs.append(output.mean(axis=1))
            input_ids.append(batch['input_ids'])
            labels.append(batch['labels'])
            attention_mask.append(batch['attention_mask'])
    
    last_layer_outputs = torch.cat(last_layer_outputs, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.cat(labels, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    sample_indexes = sample_by_cluster(last_layer_outputs.cpu().numpy(), NUM_CLUSTERS, DATA_PER_CLUSTER)
    replay_data = {'input_ids': input_ids[sample_indexes], 'labels': labels[sample_indexes], 'attention_mask': attention_mask[sample_indexes]}
    torch.save(replay_data, REPLAY_BL_PATH)
    
    
if __name__ == '__main__':
    main()