import os
import math
import umap
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
from transformer.MoMoTSwitchTailV2 import BertWithMoMoTSwitchTailV2
import matplotlib.pyplot as plt

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

# folder paths
model_name = 'wikisst-tail'
LOAD_FOLDER = "wikisst(128)300w-bs64-1epoch-lr35-momot_switch_tail_v2(common-lora)_layer(other)(3)/checkpoint-47928"
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


def get_losses(accelerator, model, data_loader):
    losses = []
    h_common, h_tail = [], []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            loss, _ = model(**batch)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'])
            if loss.item() > 6.88:
                h_tail.append(h_.cpu())
            else:
                h_common.append(h_.cpu())
            # print(i, loss.item())
            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
    return losses, h_common, h_tail



def cluster_dist(accelerator, model, data_loader, config):
    layer_cluster_list = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            for l in range(6):
                h_ = model.bert.layers.layers[l](h_, batch['attention_mask'])
            for l in range(6, 12):
                c_, _ = model.bert.layers.layers[l].routing(h_)
                counts = [len(c) for c in c_]
                if i == 0:
                    layer_cluster_list.append(counts)
                else:
                    layer_cluster_list[l-6] = [counts[c]+layer_cluster_list[l-6][c] for c in range(config.unique_experts+1)]
                h_ = model.bert.layers.layers[l](h_, batch['attention_mask'])
    return layer_cluster_list
                    
                


def get_accs(accelerator, model, data_loader):
    accs = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            loss, logits = model(**batch)
            acc = compute_accuracy(logits, batch['labels'])
            accs.append(acc)
    return accs


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    config_path = os.path.join(LOAD_PATH, 'config.json')
    config = BertConfig.from_json_file(config_path)
    
    
    """GET LOSSES"""
    dataset = WikiSST(config=config)
    base_model = BertWithMoMoTSwitchTailV2(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    base_model.load_state_dict(checkpoint)
    
    print(len(dataset.val_loader))
    # c = cluster_dist(accelerator, base_model, dataset.val_loader, config)
    # print(c)
    losses, h_common, h_tail = get_losses(accelerator, base_model, dataset.val_loader)
    print(len(h_common), len(h_tail))
    h_all = h_common + h_tail
    h_all_tensor = torch.concat(h_all, dim=0)
    
    data_reducer = umap.UMAP(random_state=42)
    h_transformed = data_reducer.fit_transform(h_all_tensor.mean(dim=1))
    h_common_transformed = h_transformed[:len(h_common)]
    h_tail_transformed = h_transformed[len(h_common):]
    
    plt.figure()
    plt.scatter(h_common_transformed[:,0], h_common_transformed[:,1], color='b', alpha=0.5, s=1)
    plt.scatter(h_tail_transformed[:,0], h_tail_transformed[:,1], color='r', alpha=0.5, s=1)
    
    plt.title("Distribution of Common & Tail")
    plt.legend()
    plt.savefig('Distribution of Common & Tail V2.png')
    
    
    # """GET ACCURACIES"""
    # # dataset_ft = SST2(config)
    # # model_ft = BertForSequenceClassification(config, num_labels=2)
    # # checkpoint = torch.load(os.path.join(LOAD_PATH_FT, 'pytorch_model.bin'))
    # # model_ft.load_state_dict(checkpoint)
    # # accs = get_accs(accelerator, model_ft, dataset_ft.val_loader)
    
    # sorted_indices = [index for index, value in sorted(enumerate(losses), key=lambda pair: pair[1])]
    # losses.sort()
    # # sorted_accs = [accs[i] for i in sorted_indices]
    # # losses = losses[sorted_indices]

    # num_intervals = 20
    # # max_loss, min_loss = losses[-1], losses[0]
    # max_loss, min_loss = 8.50, 0.00
    # interval = (max_loss - min_loss) / num_intervals
    
    # count = []
    # x_axis = []
    # stop = min_loss + interval
    # sum_stop = 0
    
    # avg_accs = []
    # total_acc = 0
    # for i, l in enumerate(losses):
    #     if l < stop:
    #         sum_stop += 1
    #         # total_acc += sorted_accs[i]
    #     else:
    #         count.append(sum_stop)
    #         x_axis.append(f'{stop-interval:.2f} ~ {stop:.2f}')
            
    #         # avg_acc = total_acc / sum_stop
    #         # avg_accs.append(avg_acc)
            
    #         total_acc = 0
    #         sum_stop = 0
    #         stop += interval
    # print(count)
    # plt.figure(figsize=(10, 6))  # Increase figure size for clarity
    # plt.bar(x_axis, count, color='blue', edgecolor='black')

    # # Improve the x-axis
    # plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and increase font size
    # # plt.xlim(min(x_axis), max(x_axis))  # Adjust x-axis limits if needed

    # plt.grid(axis='x')  # Add gridlines to x-axis for clarity

    # plt.tight_layout()  # Adjust subplot params so the subplot(s) fits in to the figure area
    # plt.savefig(f'{model_name}_loss_of_SST_VAL.png')
    
    # # plt.figure()
    # # plt.bar(x_axis, avg_accs)
    # # plt.savefig(f'{model_name} acc of SST(VAL).png')
    

if __name__ == "__main__":
    main()
    
    