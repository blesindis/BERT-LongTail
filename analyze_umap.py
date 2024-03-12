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
    # "bert",
    # "moe",
    "mot-vanilla",
    # "mot-inhib-attn"
]

center_models = {
    'mot-vanilla': "0115-bert-wiki103(256)-bs24",
    'mot-inhib-attn': "0115-mot-inhib-attn-wiki103(256)-(auto)-5000step-(fix-center-save)-(check)",
    
}

center_files = {
    'mot-vanilla': 'centers-4-vanilla(use input).pth',
    'mot-inhib-attn': 'centers.pth',
}

load_folder = {
    "bert": "0115-bert-wiki103(256)-bs24",
    "moe": "0115-moe4-wiki103(256)-bs24",
    "mot-vanilla": "0115-mot-vanilla-wiki103(256)-bs24-(save)",
    "mot-inhib-attn": "0115-mot-inhib-attn-wiki103(256)-(auto)-5000step-(fix-center-save)-(check)"
}


def get_model(model_name, load_path ,config, accelerator, center_path=None):
    if 'bert' in model_name:
        model = base_models.BertForMLM(config)
    elif 'moe' in model_name:
        model = base_models.BertSwitch(config)
    else:
        if 'vanilla' in model_name:
            centers = load_layer_data(center_path)
            model = base_models.BertWithMOE(config, centers)
        else:
            centers = torch.load(center_path, map_location='cuda')
            model = base_models.BertMOTAttnAuto(config)   
            for l in range(config.num_hidden_layers):
                model.bert.layers.layers[l].centers = centers['layer'+str(l)]
    checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    # prepare model and data on device
    model = accelerator.prepare(model) 
    return model


def get_ffn_outputs_bert(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                att_outputs = model.bert.encoders.layers[j].attention(hidden_states, batch['attention_mask'])
                ffn_outputs = model.bert.encoders.layers[j].ffn(att_outputs)
                hidden_states = model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(ffn_outputs.to('cpu'))
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], ffn_outputs.to('cpu')], dim=0)
    return layer_outputs


def get_att_outputs_bert(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                att_outputs = model.bert.encoders.layers[j].attention(hidden_states, batch['attention_mask'])
                hidden_states = model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
    return layer_outputs


def get_ffn_outputs_moe(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                att_outputs = model.bert.layers.layers[j].attention(hidden_states, batch['attention_mask'])
                ffn_outputs = model.bert.layers.layers[j].ffn(att_outputs)
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(ffn_outputs.to('cpu'))
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], ffn_outputs.to('cpu')], dim=0)
    return layer_outputs


def get_att_outputs_moe(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                att_outputs = model.bert.layers.layers[j].attention(hidden_states, batch['attention_mask'])
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(att_outputs.to('cpu'))
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
    return layer_outputs


def get_ffn_outputs_mot(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            print(i)
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
            print(i)
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


def get_ffn_outputs_mot_inhib(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].routing(hidden_states, batch['attention_mask'])
                ffn_outputs = hidden_states.new_zeros(hidden_states.shape)
                for c in range(config.num_experts):
                    att_output = model.bert.layers.layers[j].experts[c].attention(hidden_states[cluster[c],:,:], batch['attention_mask'][cluster[c], :])        
                    ffn_outputs[cluster[c],:,:] = model.bert.layers.layers[j].experts[c].ffn(att_output)
                    cluster[c] = [d + config.batch_size * i for d in cluster[c]]    
                hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(ffn_outputs.to('cpu'))
                    cluster_lists.append(cluster)
                else:
                    layer_outputs[j] = torch.cat([layer_outputs[j], ffn_outputs.to('cpu')], dim=0)
                    for c in range(config.num_experts):
                        cluster_lists[j][c] += cluster[c]                    
    return layer_outputs, cluster_lists


def get_attn_outputs_mot_inhib(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break            
            print(i)
            hidden_states = model.bert.embeddings(batch['input_ids'])
            for j in range(config.num_hidden_layers):
                cluster = model.bert.layers.layers[j].routing(hidden_states, batch['attention_mask'])
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
        load_path = os.path.join('outputs', load_folder[model_name], 'checkpoint-50000')
        center_path = None
        
        if "mot" in model_name:
            center_path = os.path.join('outputs', center_models[model_name], 'checkpoint-5000', center_files[model_name])
        
        model = get_model(model_name, load_path, config, accelerator, center_path)
        
        """Get LayerOutput"""
        if 'bert' in model_name:
            att_outputs = get_att_outputs_bert(model, train_loader, config)
            ffn_outputs = get_ffn_outputs_bert(model, train_loader, config)
        elif 'moe' in model_name:
            att_outputs = get_att_outputs_moe(model, train_loader, config)
            ffn_outputs = get_ffn_outputs_moe(model, train_loader, config)
        elif 'vanilla' in model_name:
            att_outputs, cluster_att = get_attn_outputs_mot(model, train_loader, config)
            ffn_outputs, cluster_ffn = get_ffn_outputs_mot(model, train_loader, config)
        else:
            att_outputs, cluster_att = get_attn_outputs_mot_inhib(model, train_loader, config)
            ffn_outputs, cluster_ffn = get_ffn_outputs_mot_inhib(model, train_loader, config)
        
        
        """Draw Pics"""
        colors = ['b', 'r', 'green', 'yellow']
        data_reducer = umap.UMAP(random_state=42)
        for i in range(config.num_hidden_layers):
            """Sentence"""
            # transformed_att = data_reducer.fit_transform(att_outputs[i].mean(axis=1))
            # transformed_ffn = data_reducer.fit_transform(ffn_outputs[i].mean(axis=1))
            """Token"""
            transformed_att = data_reducer.fit_transform(att_outputs[i].view(-1,768))
            transformed_ffn = data_reducer.fit_transform(ffn_outputs[i].view(-1,768))
            # plot attention output
            plt.figure()
            if 'mot' in model_name:
                for c in range(config.num_experts):
                    if len(cluster_att[i][c]):
                        """Sentence"""
                        # plt.scatter(transformed_att[cluster_att[i][c],0], transformed_att[cluster_att[i][c],1], label=str(c), color=colors[c])                
                        """Token"""
                        list_ = []
                        for b in range(config.batch_size):
                            list_ += [value + b for value in cluster_att[i][c]]
                        plt.scatter(transformed_att[list_,0], transformed_att[list_,1], label=str(c), color=colors[c], alpha=0.5)                
            else:
                plt.scatter(transformed_att[:,0], transformed_att[:,1])
            plt.title("Layer " + str(i) + " Attention Output")
            plt.legend()
            plt.savefig('Layer ' + str(i) + ' Attention Output of ' + model_name + '.png')
            
            # plot ffn output
            plt.figure()
            if 'mot' in model_name:
                for c in range(config.num_experts):
                    if len(cluster_att[i][c]):
                        """Sentence"""
                        # plt.scatter(transformed_ffn[cluster_ffn[i][c], 0], transformed_ffn[cluster_ffn[i][c], 1], label=str(c), color=colors[c])
                        """Token"""
                        list_ = []
                        for b in range(config.batch_size):
                            list_ += [value + b for value in cluster_ffn[i][c]]
                        plt.scatter(transformed_ffn[list_, 0], transformed_ffn[list_, 1], label=str(c), color=colors[c], alpha=0.5, s=5)
            else:
                plt.scatter(transformed_ffn[:,0], transformed_ffn[:,1])
            plt.title("Layer " + str(i) + " FFN Output")
            plt.legend()
            plt.savefig('Layer ' + str(i) + ' FFN Output of ' + model_name + '.png')
        

if __name__ == "__main__":
    main()
    
    