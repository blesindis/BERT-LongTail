import os
import umap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import MixedPretrain, ACLForLM, RestaurantForLM, Wikitext103
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)


SAMPLE_BATCHES = 10
NUM_EXPERTS = 2

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

CONFIG_PATH = 'config/bert_a.json'

model_name = 'bert'
load_folder = "bert(128)300w-bs64-1epoch-lr1-bert/checkpoint-30000"



def get_attn_outputs(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            hidden_states = model.bert.embeddings(batch['input_ids'])

            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                hidden_states = model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(hidden_states.to('cpu'))
                else:
                    layer_outputs[j] = torch.cat((layer_outputs[j], hidden_states.to('cpu')), dim=0)
    return layer_outputs


def get_layer_outputs(model, train_loader, config, center_model):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            
            routing_states = []             
            hidden_states = center_model.bert.embeddings(batch['input_ids'])
            routing_states.append(hidden_states)                    
            for j in range(config.num_hidden_layers):
                hidden_states = center_model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                routing_states.append(hidden_states)
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            if i == 0:
                layer_outputs.append(hidden_states.to('cpu'))
            else:
                layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].attention.routing(routing_states[j])
                print([len(cluster[_]) for _ in range(len(cluster))])
                for c in range(NUM_EXPERTS):
                    cluster[c] = [d.to('cpu') + config.batch_size * i for d in cluster[c]]                
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'], routing_states[j])
                
                if i == 0:
                    layer_outputs.append(hidden_states.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
                    for c in range(NUM_EXPERTS):
                        cluster_lists[j][c] += cluster[c]
    return layer_outputs, cluster_lists


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    # dataset = RestaurantForLM(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    # LOAD DATASET
    dataset = MixedPretrain(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    
    # LOAD MODEL
    # load_path = os.path.join('outputs', load_folder, 'checkpoint-15000')
    load_path = os.path.join('outputs', load_folder)
    
    checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
    model = base_models.BertForMLM(config)
    model.load_state_dict(checkpoint)
    model = accelerator.prepare(model) 
    
    # GET OUTPUTS
    layer_outputs = get_attn_outputs(model, train_loader, config)
    # layer_outputs, clusters = get_attn_outputs_by_cluster(model, train_loader, config, center_model)
    
    # DRAW PICS
    colors = ['b', 'r', 'green', 'yellow']
    data_reducer = umap.UMAP(random_state=42)
    for i in range(config.num_hidden_layers):
        o = layer_outputs[i].mean(axis=1)
        transformed_att = data_reducer.fit_transform(o)
        # center_i = transformed_att[:NUM_EXPERTS]
        # output_i = transformed_att[NUM_EXPERTS:]
        output_i = transformed_att
        plt.figure()
        # plt.scatter(center_i[:,0], center_i[:,1], label='yellow', s=100)
        plt.scatter(output_i[:,0], output_i[:,1])
        plt.title("Layer " + str(i) + " Attention Output")
        plt.legend()
        plt.savefig('Layer ' + str(i) + ' Attention Output of ' + model_name + '.png')
        

if __name__ == "__main__":
    main()
    