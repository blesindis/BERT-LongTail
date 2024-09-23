import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from transformer.FT_BERT import BertForSequenceClassification
from transformer.BERT import BertForMLM
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from Dataset_ft_pure import SST2_pure
from Dataset_ft import SST2
from Dataset import Wikipedia, WikiSST
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    copy_parameters
)

SAMPLE_BATCHES = 20

# folder paths
model_name = 'wiki-bert'
LOAD_FOLDER = "wiki(128)300w-bs64-1epoch-lr1-bert"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CONFIG_PATH = 'config/bert_a.json'


colors = [
    '#FF0000',  # Red
    '#008000',  # Green
    '#0000FF',  # Blue
    '#00FFFF',  # Cyan
    '#FF00FF',  # Magenta
    '#FFFF00',  # Yellow
    '#000000',  # Black
    '#008080',  # Teal
    '#FFA500',  # Orange
    '#800080',  # Purple
    '#A52A2A',  # Brown
    '#00FF00'   # Lime
]


def get_layer_outputs(accelerator, model, data_loader):
    layer_outputs = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= SAMPLE_BATCHES:
                break
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(h_.to('cpu'))
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], h_.to('cpu')], dim=0)
            
    return layer_outputs


def get_attn_outputs(accelerator, model, data_loader):
    layer_outputs = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= SAMPLE_BATCHES:
                break
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                att_ = model.bert.encoders.layers[l].attention(h_, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(att_.to('cpu'))
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], att_.to('cpu')], dim=0)
            
    return layer_outputs    


def get_sorted_abs(tensors):
    abs_tensors = torch.abs(tensors)
    sum_abs_tensors = torch.sum(abs_tensors, dim=0)
    mean_abs_tensors = sum_abs_tensors / len(tensors)
    sorted_mean_abs_tensors, indices = torch.sort(mean_abs_tensors, descending=True)
    return sorted_mean_abs_tensors, indices


def get_sorted_var(tensors):
    variance_tensors = torch.var(tensors, dim=0, unbiased=True)
    sorted_variance_tensors, indices = torch.sort(variance_tensors, descending=True)
    return sorted_variance_tensors, indices


def get_cossim_mat(tensors):
    # Normalize each tensor to have unit norm
    norms = torch.norm(tensors, p=2, dim=1, keepdim=True)
    normalized_tensors = tensors / norms

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(normalized_tensors, normalized_tensors.t())
    return cosine_similarity_matrix


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    """GET LOSSES"""
    dataset = Wikipedia(config=config)
    train_loader = dataset.train_loader
    
    index = 0
    while index < 46000:
        load_path = os.path.join(LOAD_PATH, f'checkpoint-{index}')
    
        base_model = BertForMLM(config)
        checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
        base_model.load_state_dict(checkpoint)
    
        layer_outputs = get_layer_outputs(accelerator, base_model, train_loader)
        # layer_outputs = get_attn_outputs(accelerator, base_model, train_loader)
        
        # PLOT Token Within Seq Cos Sim Mat
        for l, outputs in enumerate(layer_outputs):
            token_mat = None
            for count, o in enumerate(outputs):
                mat = get_cossim_mat(o)
                if count == 0:
                    token_mat = mat
                else:
                    token_mat += mat
            token_mat /= len(outputs)
            cosine_similarity_matrix_np = token_mat.numpy()
            plt.figure()
            sns.heatmap(cosine_similarity_matrix_np, cmap='viridis')
            plt.title('Cosine Similarity Matrix Heatmap')
            plt.xlabel('Tensor Index')
            plt.ylabel('Tensor Index')

            plt.legend()
            plt.savefig(f'Dim Var of checkpoint{index} of {model_name} at layer {l} (Token-Sequence).png')
        
        
        
        # # PLOT Token Within Sequence VAR
        # plt.figure()
        # for l, outputs in enumerate(layer_outputs):
        #     tokens = None
        #     for count, o in enumerate(outputs):
        #         print(o.shape)
        #         sorted_o, _ = get_sorted_var(o)
        #         if count == 0:
        #             tokens = sorted_o
        #         else:
        #             tokens += sorted_o
        #     tokens /= len(outputs)
        #     plt.plot(tokens.numpy(), label=f'layer-{l}', c=colors[l])
        # plt.title('Sorted Token-Sequence Dim-wise Variance Value')
        # plt.xlabel('Dimension')
        # plt.ylabel('Variance')
        # plt.legend()
        # plt.savefig(f'Dim Var of checkpoint{index} of {model_name} (Token-Sequence).png')
        
        # # PLOT ABS
        # plt.figure()
        # for l, outputs in enumerate(layer_outputs):
        #     # sorted_abs = get_sorted_abs(outputs.view(-1, 768))
        #     sorted_abs, indices = get_sorted_abs(outputs.mean(dim=1))
        #     plt.plot(sorted_abs.numpy(), label=f'layer-{l}', c=colors[l])
        # plt.title('Sorted Dim-wise Absolute Value')
        # plt.xlabel('Dimension')
        # plt.ylabel('Absolute Value')
        # plt.legend()
        # plt.savefig(f'Dim Abs of checkpoint{index} of {model_name} (Sequence) at Attn.png')
        
        # # PLOT VAR
        # plt.figure()
        # for l, outputs in enumerate(layer_outputs):
        #     # sorted_abs = get_sorted_var(outputs.view(-1, 768))
        #     sorted_abs, indices = get_sorted_var(outputs.mean(dim=1))
        #     if l == 11:
        #         print(f'checkpoint-{index}: ', indices[-10:])
        #     plt.plot(sorted_abs.numpy(), label=f'layer-{l}', c=colors[l])
        # plt.title('Sorted Dim-wise Variance Value')
        # plt.xlabel('Dimension')
        # plt.ylabel('Variance')
        # plt.legend()
        # plt.savefig(f'Dim Var of checkpoint{index} of {model_name} (Sequence) at Attn.png')
        
        print(f'Checkpoint {index} Finished Processing')
        del base_model
        
        index += 5000
        
        
    

if __name__ == "__main__":
    main()
    
    