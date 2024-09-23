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
from transformer.MoMoTModelRouterMTL import BertWithMoMoTModelRouterMTL
import matplotlib.pyplot as plt

# Local imports
from Dataset_ft_pure import SST2_pure
from Dataset_ft import SST2
from Dataset import Wikipedia, WikiSST
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    copy_parameters,
    load_layer_data_last
)
NUM_EXPERTS = 6
# folder paths
model_name = 'mtl'
dataset_name = 'wiki'
CENTER_MODEL_PATH = "outputs/wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, f'centers-{NUM_EXPERTS}-momoe-transformer-lastlayer.pth')
LOAD_FOLDER = "wiki(128)300w-bs64-1epoch-lr3-momot_model_router_mtl(lora-full-384, 4-4, att-ogd)_layer(full)_router5000/checkpoint-46875"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
LOAD_FOLDER_FT = "ft-sst-wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
LOAD_PATH_FT = os.path.join('outputs', LOAD_FOLDER_FT)
CONFIG_PATH = 'config/bert_bs1.json'

# training parameters
num_epochs = 1
lr = 1e-4
weight_decay = 0.01


def compute_accuracy(logits, labels):
    # Convert logits to probabilities and then to predicted class indexes
    probs = torch.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probs, dim=1)

    # Compare with true labels to find how many predictions were correct
    correct_predictions = (predicted_labels == labels).sum().item()

    # Calculate accuracy
    accuracy = correct_predictions / labels.size(0)  # labels.size(0) gives the batch size

    return accuracy


def get_losses(accelerator, model, data_loader, center_model, centers):
    losses = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            _, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]   
            
            loss, _ = model(**batch, cluster_list=cluster_list)
            # print(i, loss.item())
            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
    return losses


def get_accs(accelerator, model, data_loader):
    accs = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            loss, logits = model(**batch)
            acc = compute_accuracy(logits, batch['labels'])
            print(acc)
            accs.append(acc)
    return accs


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    config_path = os.path.join(LOAD_PATH, 'config.json')
    config = BertConfig.from_json_file(config_path)
    
    """GET LOSSES"""
    center_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    center_model = accelerator.prepare(center_model)
    
    centers = load_layer_data_last(CENTER_PATH)
    
    dataset = Wikipedia(config=config)
    base_model = BertWithMoMoTModelRouterMTL(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    base_model.load_state_dict(checkpoint)
    
    print(len(dataset.val_loader))
    losses = get_losses(accelerator, base_model, dataset.val_loader, center_model, centers)
    
    """GET ACCURACIES"""
    # dataset_ft = SST2(config)
    # model_ft = BertForSequenceClassification(config, num_labels=2)
    # checkpoint = torch.load(os.path.join(LOAD_PATH_FT, 'pytorch_model.bin'))
    # model_ft.load_state_dict(checkpoint)
    # accs = get_accs(accelerator, model_ft, dataset_ft.val_loader)
    
    sorted_indices = [index for index, value in sorted(enumerate(losses), key=lambda pair: pair[1])]
    losses.sort()
    # sorted_accs = [accs[i] for i in sorted_indices]
    # losses = losses[sorted_indices]

    num_intervals = 20
    # max_loss, min_loss = losses[-1], losses[0]
    max_loss, min_loss = 12.50, 0.00
    interval = (max_loss - min_loss) / num_intervals
    
    
    count = []
    x_axis = []
    stop = min_loss + interval
    sum_stop = 0
    
    avg_accs = []
    total_acc = 0
    for i, l in enumerate(losses):
        if l < stop:
            sum_stop += 1
            # total_acc += sorted_accs[i]
        else:
            count.append(sum_stop)
            x_axis.append(f'{stop-interval:.2f} ~ {stop:.2f}')
            
            # avg_acc = total_acc / sum_stop
            # avg_accs.append(avg_acc)
            
            total_acc = 0
            sum_stop = 0
            stop += interval
    print(x_axis, count)
    plt.figure(figsize=(10, 6))  # Increase figure size for clarity
    plt.bar(x_axis, count, color='blue', edgecolor='black')

    # Improve the x-axis
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and increase font size
    # plt.xlim(min(x_axis), max(x_axis))  # Adjust x-axis limits if needed

    plt.grid(axis='x')  # Add gridlines to x-axis for clarity

    # plt.tight_layout()  # Adjust subplot params so the subplot(s) fits in to the figure area
    plt.savefig(f'{model_name}_loss_of_{dataset_name}_VAL.png')
    
    # plt.figure()
    # plt.bar(x_axis, avg_accs)
    # plt.savefig(f'{model_name} acc of SST(VAL).png')
    

if __name__ == "__main__":
    main()
    
    