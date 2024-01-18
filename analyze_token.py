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
from Dataset import MixedData, ACLForLM, RestaurantForLM, Wikitext103
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)

SAMPLE_BATCHES = 10

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

CONFIG_PATH = 'config/bert_256.json'

model_names = [
    # "mot-input-token",
    # "mot-input-sentence",
    # "mot-gating-token",
    "mot-gating-sentence"
]

center_models = {
    'mot-input-sentence': "0115-bert-wiki103(256)-bs24/checkpoint-5000",
    'mot-input-token': "0112-bert-wiki103(256)/checkpoint-10000",
    
}

center_files = {
    'mot-input-sentence': 'centers-4-vanilla(use input).pth',
    'mot-input-token': 'centers-4-vanilla(use input)-token.pth',
}

load_folder = {
    "mot-input-token": "0117-mot-vanilla-wiki103(256)-bs24-(save)-token",
    "mot-input-sentence": "0115-mot-vanilla-wiki103(256)-bs24-(save)",
    "mot-gating-token": "0117-moet-switch-wiki103(256)-bs24-(save)-token",
    "mot-gating-sentence": "0117-moet-switch-wiki103(256)-bs24-(save)-sentence"
}


def get_model(model_name, load_path ,config, accelerator, center_path=None):
    if 'input' in model_name:
        centers = load_layer_data(center_path)
        model = base_models.BertWithMOE(config,centers)
    else:
        model = base_models.BertMoTSwitch(config)
    
    checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    # prepare model and data on device
    model = accelerator.prepare(model) 
    return model


def get_ffn_outputs_mot(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].routing(hidden_states)
                ffn_outputs = hidden_states.new_zeros(hidden_states.shape)
                for c in range(config.num_experts):
                    att_output = model.bert.layers.layers[j].experts[c].attention(hidden_states[cluster[c],:,:], batch['attention_mask'][cluster[c], :])        
                    ffn_outputs[cluster[c],:,:] = model.bert.layers.layers[j].experts[c].ffn(att_output)
                    cluster[c] = [d +config.batch_size * i for d in cluster[c]]    
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(ffn_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], ffn_outputs.to('cpu')], dim=0)
                    for c in range(config.num_experts):
                        cluster_lists[j][c] += cluster[c]                    
    return layer_outputs, cluster_lists


def get_attn_outputs_mot(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].routing(hidden_states)
                print(len(cluster[0]), len(cluster[1]))
                att_outputs = hidden_states.new_zeros(hidden_states.shape)
                for c in range(config.num_experts):
                    h_ = model.bert.layers.layers[j].experts[c].attention.self(hidden_states[cluster[c],:,:], batch['attention_mask'][cluster[c],:])
                    att_outputs[cluster[c],:,:] = model.bert.layers.layers[j].experts[c].attention.dense(h_)
                    cluster[c] = [d + config.batch_size * i for d in cluster[c]]                
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
                    for c in range(config.num_experts):
                        cluster_lists[j][c] += cluster[c]
    return layer_outputs, cluster_lists


# def get_ffn_outputs_mot_token(model, train_loader, config):
#     with torch.no_grad():
#         layer_outputs = []
#         cluster_lists = []
#         for i, batch in enumerate(train_loader):
#             if i >= SAMPLE_BATCHES:
#                 break            
#             hidden_states = model.bert.embeddings(batch['input_ids'])
#             for j in range(config.num_hidden_layers):
#                 cluster = model.bert.layers.layers[j].routing(hidden_states)
#                 ffn_outputs = hidden_states.new_zeros(hidden_states.shape)
#                 for c in range(config.num_experts):
#                     att_output = model.bert.layers.layers[j].experts[c].attention(hidden_states[cluster[c],:,:], batch['attention_mask'][cluster[c], :])        
#                     ffn_outputs[cluster[c],:,:] = model.bert.layers.layers[j].experts[c].ffn(att_output)
#                     cluster[c] = [d +config.batch_size * i for d in cluster[c]]    
#                 hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
#                 if i == 0:
#                     layer_outputs.append(ffn_outputs.to('cpu'))
#                     cluster_lists.append(cluster)
#                 else:
#                     layer_outputs[j] = torch.cat([layer_outputs[j], ffn_outputs.to('cpu')], dim=0)
#                     for c in range(config.num_experts):
#                         cluster_lists[j][c] += cluster[c]                    
#     return layer_outputs, cluster_lists


def get_attn_outputs_mot_token(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].routing(hidden_states)
                print([len(cl) for cl in cluster])
                h_ = hidden_states.view(-1, 768)
                att_outputs = h_.new_zeros(h_.shape)
                for c in range(config.num_experts):
                    if len(cluster[c]):
                        att = model.bert.layers.layers[j].experts[c].attention.self(h_[cluster[c], :].unsqueeze(1), batch['attention_mask'].view(h_.shape[0], -1)[cluster[c]]).view(len(cluster[c]), -1)
                        att_outputs[cluster[c],:] = model.bert.layers.layers[j].experts[c].attention.dense(att).view(-1, 768)
                        cluster[c] = [d + config.batch_size * i for d in cluster[c]]                
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
                    for c in range(config.num_experts):
                        cluster_lists[j][c] += cluster[c]
    return layer_outputs, cluster_lists


def get_attn_outputs_moe_token(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                h_ = hidden_states.view(-1, 768)
                route_prob = model.bert.layers.layers[j].softmax( model.bert.layers.layers[j].switch(h_))
                route_prob_max, routes = torch.max(route_prob, dim=-1)
                cluster = [torch.eq(routes, i).nonzero(as_tuple=True)[0].cpu().numpy() for i in range(config.num_experts)]      
                print([len(cl) for cl in cluster])
                att_outputs = h_.new_zeros(h_.shape)
                for c in range(config.num_experts):
                    if len(cluster[c]):
                        att = model.bert.layers.layers[j].experts[c].attention.self(h_[cluster[c], :].unsqueeze(1), batch['attention_mask'].view(h_.shape[0], -1)[cluster[c]]).view(len(cluster[c]), -1)
                        att_outputs[cluster[c],:] = model.bert.layers.layers[j].experts[c].attention.dense(att).view(-1, 768)
                        cluster[c] = [d + config.batch_size * i for d in cluster[c]]                
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
                    for c in range(config.num_experts):
                        cluster_lists[j][c] += cluster[c]
    return layer_outputs, cluster_lists


def get_attn_outputs_moe_sentence(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                route_prob = model.bert.layers.layers[j].softmax( model.bert.layers.layers[j].switch(hidden_states.mean(dim=1)))
                route_prob_max, routes = torch.max(route_prob, dim=-1)
                cluster = [torch.eq(routes, i).nonzero(as_tuple=True)[0].cpu().numpy() for i in range(config.num_experts)]      
                print([len(cl) for cl in cluster])
                att_outputs = hidden_states.new_zeros(hidden_states.shape)
                for c in range(config.num_experts):
                    if len(cluster[c]):
                        att = model.bert.layers.layers[j].experts[c].attention.self(hidden_states[cluster[c], :, :], batch['attention_mask'][cluster[c], :])
                        att_outputs[cluster[c], : ,:] = model.bert.layers.layers[j].experts[c].attention.dense(att)
                        cluster[c] = [d + config.batch_size * i for d in cluster[c]]                
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
                    for c in range(config.num_experts):
                        cluster_lists[j][c] += cluster[c]
        layer_outputs = [l.mean(dim=1) for l in layer_outputs]
    return layer_outputs, cluster_lists


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    # dataset = RestaurantForLM(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    dataset = Wikitext103(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    """Load Model"""
    for model_name in model_names:        
        load_path = os.path.join('outputs', load_folder[model_name], 'checkpoint-10000')
        center_path = None
        
        if "input" in model_name:
            center_path = os.path.join('outputs', center_models[model_name], center_files[model_name])
        
        model = get_model(model_name, load_path, config, accelerator, center_path)
        
        """Get LayerOutput"""
        if 'input' in model_name:
            if 'token' in model_name:
                att_outputs, cluster_att = get_attn_outputs_mot_token(model, train_loader, config)
        else:
            if 'token' in model_name:
                att_outputs, cluster_att = get_attn_outputs_moe_token(model, train_loader, config)
            else:
                att_outputs, cluster_att = get_attn_outputs_moe_sentence(model, train_loader, config)
        
        
        """Draw Pics"""
        colors = ['b', 'r', 'green', 'yellow']
        data_reducer = umap.UMAP(random_state=42)
        for i in range(config.num_hidden_layers):
            """Sentence"""
            # transformed_att = data_reducer.fit_transform(att_outputs[i].mean(axis=1))
            # transformed_ffn = data_reducer.fit_transform(ffn_outputs[i].mean(axis=1))
            """Token"""
            transformed_att = data_reducer.fit_transform(att_outputs[i])
            # transformed_ffn = data_reducer.fit_transform(ffn_outputs[i].view(-1,768))
            # plot attention output
            plt.figure()
            if 'mot' in model_name:
                for c in range(config.num_experts):
                    if len(cluster_att[i][c]):
                        """Sentence"""
                        plt.scatter(transformed_att[cluster_att[i][c],0], transformed_att[cluster_att[i][c],1], label=str(c), color=colors[c])                
                        """Token"""
                        # list_ = []
                        # for b in range(config.batch_size):
                        #     list_ += [value + b for value in cluster_att[i][c]]
                        # plt.scatter(transformed_att[list_,0], transformed_att[list_,1], label=str(c), color=colors[c], alpha=0.5)                
            else:
                plt.scatter(transformed_att[:,0], transformed_att[:,1])
            plt.title("Layer " + str(i) + " Attention Output")
            plt.legend()
            plt.savefig('Layer ' + str(i) + ' Attention Output of ' + model_name + '.png')
            
            # # plot ffn output
            # plt.figure()
            # if 'mot' in model_name:
            #     for c in range(config.num_experts):
            #         if len(cluster_att[i][c]):
            #             """Sentence"""
            #             # plt.scatter(transformed_ffn[cluster_ffn[i][c], 0], transformed_ffn[cluster_ffn[i][c], 1], label=str(c), color=colors[c])
            #             """Token"""
            #             list_ = []
            #             for b in range(config.batch_size):
            #                 list_ += [value + b for value in cluster_ffn[i][c]]
            #             plt.scatter(transformed_ffn[list_, 0], transformed_ffn[list_, 1], label=str(c), color=colors[c], alpha=0.5, s=5)
            # else:
            #     plt.scatter(transformed_ffn[:,0], transformed_ffn[:,1])
            # plt.title("Layer " + str(i) + " FFN Output")
            # plt.legend()
            # plt.savefig('Layer ' + str(i) + ' FFN Output of ' + model_name + '.png')
        

if __name__ == "__main__":
    main()
    
    