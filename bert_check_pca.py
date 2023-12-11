"""
Check the distribution of path, i.e., for each path, how many data are assigned to it
"""
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import BertConfig

# Local imports
import base_models
from utils.pic_utils import pca_scatter
from utils.metric_utils import DBI
from utils.math_utils import (
    pca_fix, 
    pca_auto,
)
from Dataset import RestaurantForLM, ACLForLM

# cluster config
SAMPLE_BATCHES = 20

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder path
PIC_NAME = "1127-bert-joint-lr1.5-100epoch-checkpoint-50"
LOAD_FOLDER = "1127-bert-joint-lr1.5-100epoch/checkpoint-50"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CONFIG_PATH = 'config/bert_a.json'


def main():  
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset1 = RestaurantForLM(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    dataset2 = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    
    model = base_models.BertForMLM(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader1, train_loader2 = dataset1.train_loader, dataset2.train_loader
    
    accelerator = Accelerator()
    
    model, train_loader1, train_loader2 = accelerator.prepare(model, train_loader1, train_loader2)

    # get last layer outputs of 2 dataset
    last_layer_outputs1 = []    
    last_layer_outputs2 = []
    with torch.no_grad():
        for i, (batch1, batch2) in enumerate(zip(train_loader1, train_loader2)):         
            if i >= SAMPLE_BATCHES: 
                break
            
            last_layer_output1 = model.bert(batch1['input_ids'], batch1['attention_mask'])
            last_layer_output2 = model.bert(batch2['input_ids'], batch2['attention_mask'])
            
            last_layer_outputs1.append(last_layer_output1)
            last_layer_outputs2.append(last_layer_output2)
                    
    last_layer_outputs1 = torch.cat(last_layer_outputs1, dim=0)   
    last_layer_outputs2 = torch.cat(last_layer_outputs2, dim=0)     
    
    pca_outputs = pca_auto(torch.cat((last_layer_outputs1, last_layer_output2), dim=0), 0.2)
    pca_outputs1, pca_outputs2 = pca_outputs[:len(last_layer_outputs1)], pca_outputs[len(last_layer_outputs1):]   
    pca_dbi_20 = DBI(pca_outputs1, pca_outputs2)
    
    pca_outputs = pca_auto(torch.cat((last_layer_outputs1, last_layer_output2), dim=0), 0.5)
    pca_outputs1, pca_outputs2 = pca_outputs[:len(last_layer_outputs1)], pca_outputs[len(last_layer_outputs1):]     
    pca_dbi_50 = DBI(pca_outputs1, pca_outputs2)
    
    pca_outputs = pca_auto(torch.cat((last_layer_outputs1, last_layer_output2), dim=0), 0.8)
    pca_outputs1, pca_outputs2 = pca_outputs[:len(last_layer_outputs1)], pca_outputs[len(last_layer_outputs1):]     
    pca_dbi_80 = DBI(pca_outputs1, pca_outputs2)
    # calculate DBI
    dbi = DBI(last_layer_outputs1.mean(dim=1), last_layer_outputs2.mean(dim=1))    
    print(dbi, pca_dbi_20, pca_dbi_50, pca_dbi_80)
    
    # get 2-dim pca scatter pic
    pca_scatter(last_layer_outputs1, last_layer_outputs2, PIC_NAME)
        
        
    
if __name__ == '__main__':
    main()