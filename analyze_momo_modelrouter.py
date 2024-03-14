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
from transformer.MoMoModelRouter import BertWithMoMoModelRouter
from transformer.MoMoModelRouterCommonLargeNew import BertWithMoMoModelRouterCommonAttnLargeNew
from transformer.BERT import BertForMLM

# Local imports
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

model_name = 'momo-modelrouter'

center_model_path = "bert(128)300w-bs64-1epoch-lr1-bert/checkpoint-10000"

center_file = 'centers-2-momoe-transformer.pth'

load_folder = "bert(128)300w-bs64-1epoch-lr3-momo_model_router_common_lora128_router10000/checkpoint-30000"



def get_attn_outputs_by_cluster(model, train_loader, config, center_model):
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
            for j in range(config.num_hidden_layers):
                outputs_by_cluster = []
                
                cluster = model.bert.layers.layers[j].attention.routing(routing_states[j])
                print([len(cluster[_]) for _ in range(len(cluster))])
                
                for c in range(NUM_EXPERTS):
                    outputs_by_cluster.append(model.bert.layers.layers[j].attention.experts[c](hidden_states[cluster[c],:,:], batch['attention_mask'][cluster[c],:]).to('cpu'))
                    cluster[c] = [d.to('cpu') + config.batch_size * i for d in cluster[c]]   
                                  
                outputs_by_cluster.append(model.bert.layers.layers[j].attention.common_expert(hidden_states, batch['attention_mask']).to('cpu'))
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'], routing_states[j])
                
                if i == 0:
                    layer_outputs.append(outputs_by_cluster)
                    cluster_lists.append(cluster)
                else:
                    for m in range(len(outputs_by_cluster)):
                        layer_outputs[j][m] = torch.cat([layer_outputs[j][m], outputs_by_cluster[m]], dim=0)
                    for c in range(NUM_EXPERTS):
                        cluster_lists[j][c] += cluster[c]
    return layer_outputs, cluster_lists


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


def get_attn_outputs(model, train_loader, config, center_model):
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
            # if i == 0:
            #     layer_outputs.append(hidden_states.to('cpu'))
            # else:
            #     layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].attention.routing(routing_states[j])
                print([len(cluster[_]) for _ in range(len(cluster))])
                for c in range(NUM_EXPERTS):
                    cluster[c] = [d.to('cpu') + config.batch_size * i for d in cluster[c]]                
                att_outputs = model.bert.layers.layers[j].attention(hidden_states, batch['attention_mask'], routing_states[j])
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'], routing_states[j])
                
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
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
    center_path = os.path.join('outputs', center_model_path, center_file)
    
    center_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join('outputs', center_model_path, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    center_model = accelerator.prepare(center_model)
    
    centers = load_layer_data(center_path)
    checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
    model = BertWithMoMoModelRouterCommonAttnLargeNew(config, centers)
    model.load_state_dict(checkpoint)
    model = accelerator.prepare(model) 
    
    # GET OUTPUTS
    # layer_outputs, clusters = get_attn_outputs(model, train_loader, config, center_model)
    layer_outputs, clusters = get_attn_outputs_by_cluster(model, train_loader, config, center_model)
    
    # DRAW PICS
    colors = ['b', 'r', 'green', 'yellow']
    data_reducer = umap.UMAP(random_state=42)
    for i in range(config.num_hidden_layers):
        
        """Cluster Outputs of Common & Unique (Without Add)"""
        cluster_outputs = layer_outputs[i]
        concat_outputs = torch.cat(cluster_outputs, dim=0)
        transformed_o = data_reducer.fit_transform(concat_outputs.mean(axis=1))
        
        clusters = []
        start = 0
        for e in range(len(cluster_outputs)-1):
            clusters.append(transformed_o[start: start + len(cluster_outputs[e])])
            start += len(cluster_outputs[e])
        clusters.append(transformed_o[start:])
        plt.figure()
        print(len(clusters))
        for c in range(NUM_EXPERTS+1):
            plt.scatter(clusters[c][:,0], clusters[c][:,1], label=str(c), color=colors[c])                
            
        plt.title("Layer " + str(i) + " Attention Output")
        plt.legend()
        plt.savefig('Layer ' + str(i) + ' Attention Output of ' + model_name + '.png')
        
        """Attention Outputs distincted by Experts"""
        # # o = torch.cat([centers[i].to('cpu'), layer_outputs[i].mean(axis=1)], dim=0)
        # o = layer_outputs[i].mean(axis=1)
        # transformed_att = data_reducer.fit_transform(o)
        # # center_i = transformed_att[:NUM_EXPERTS]
        # # output_i = transformed_att[NUM_EXPERTS:]
        # output_i = transformed_att
        # plt.figure()
        # # plt.scatter(center_i[:,0], center_i[:,1], label='yellow', s=100)
        # for c in range(NUM_EXPERTS+1):
        #     if len(clusters[i][c]):
                
        #         plt.scatter(output_i[clusters[i][c],0], output_i[clusters[i][c],1], label=str(c), color=colors[c])                
        # plt.title("Layer " + str(i) + " Attention Output")
        # plt.legend()
        # plt.savefig('Layer ' + str(i) + ' Attention Output of ' + model_name + '.png')
        

if __name__ == "__main__":
    main()
    