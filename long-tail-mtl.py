import os
import math
import random
import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from transformer.BERT import BertForMLM
from transformer.MoMoTModelRouterMTL import BertWithMoMoTModelRouterMTL
import matplotlib.pyplot as plt

# Local imports
from Dataset import BERTPretrain, LePuWiki, Wikipedia, WikiSST
from utils.sample_utils import *
from utils.train_utils import (
    set_seed,
    load_layer_data_last,
)

from collections import Counter

NUM_EXPERTS = 4
NUM_SAMPLES = 5000
# folder paths
model_name = 'wiki-mtl'
CENTER_MODEL_PATH = "outputs/wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, f'centers-{NUM_EXPERTS}-momoe-transformer-lastlayer.pth')
LOAD_FOLDER = "wiki(128)300w-bs64-1epoch-lr3-momot_model_router_mtl(lora-full-384, 4-4, att-ogd)_layer(full)_router5000/checkpoint-46875"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CONFIG_PATH = 'config/mtl.json'

# Assuming 'lm_datasets_train' is loaded and contains tokenized data
def calculate_word_frequency(dataset, tokenizer):
    word_freq = Counter()

    # Iterate through the dataset
    for batch in dataset:
        # Each 'input_ids' in the batch represents tokenized text
        input_ids = batch['input_ids']

        # Decode tokens to words
        words = [tokenizer.decode(token_ids) for token_ids in input_ids]

        # Update word frequency count
        for word in words:
            word_freq.update(word.split())

    return word_freq


def score_data(data_loader, tokenizer, word_freq):
    scores = []
    for i, batch in enumerate(data_loader):
        if i > NUM_SAMPLES:
            break
        input_ids = batch['input_ids']
        words = [tokenizer.decode(token_ids) for token_ids in input_ids]
        
        total_freq = 0
        valid_word = 0
        for word in words:
            word = word.strip()
            if word in word_freq:            
                total_freq += word_freq[word.strip()]
                valid_word += 1
        
        average_freq = total_freq / valid_word
        scores.append(average_freq)
    return scores


def loss_dist(accelerator, model, data_loader, center_model, centers):
    losses = []
    dists = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES:
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            min_dist, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]  
            
            loss, _ = model(**batch, cluster_list=cluster_list)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)

            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
            dists.append(min_dist.cpu().item())
        
    return losses, dists


def loss_dist_new_center(accelerator, model, data_loader, center_model, centers):
    losses = []
    cluster_outputs = [[] for _ in range(NUM_EXPERTS)]
    cluster_centers = []
    dists = []
    
    # PREPARE FOR GET NEW CENTER
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES:
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            min_dist, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]  
            
            loss, _ = model(**batch, cluster_list=cluster_list)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)
            for index, c in enumerate(cluster_list):
                if len(c):
                    cluster_outputs[index].append(h_.mean(dim=1))

            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
    
    # GET NEW CENTER
    for c in cluster_outputs:
        if len(c):
            cluster_centers.append(sum(c) / len(c))
        else:
            cluster_centers.append(0)
        
    # GET NEW DISTS
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES:
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            min_dist, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]  
            
            index_ = 0
            for index, c in enumerate(cluster_list):
                if len(c):
                    index_ = index
                    break
            loss, _ = model(**batch, cluster_list=cluster_list)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)
            min_dist = torch.cdist(h_.mean(dim=1), cluster_centers[index_])

            if math.isnan(loss.item()):
                continue
            dists.append(min_dist.item())
        
    return losses, dists


def avg(l):
    return sum(l) / len(l)


def main():
    accelerator = Accelerator()
    config_path = os.path.join(LOAD_PATH, 'config.json')
    config = BertConfig.from_json_file(config_path)
    
    dataset = Wikipedia(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    center_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    center_model = accelerator.prepare(center_model)
    
    centers = load_layer_data_last(CENTER_PATH)
    
    base_model = BertWithMoMoTModelRouterMTL(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    base_model.load_state_dict(checkpoint)
        
    # Define the path to your CSV file
    file_path = 'word_frequencies.json'

    # Create a dictionary to store the words and their frequencies
    # word_frequencies = {}
    
    with open(file_path, 'r') as file:
        word_frequencies = json.load(file)
    
    # # Open the CSV file and read it
    # with open(file_path, 'r', encoding='utf-8') as csvfile:
    #     reader = csv.reader(csvfile)
    #     next(reader)  # Skip the header row
    #     for row in reader:
    #         word, frequency = row
    #         word_frequencies[word] = int(frequency)  # Convert frequency back to an integer

    # Get Data Freq
    scores = score_data(dataset.train_set, dataset.tokenizer, word_frequencies)
    # losses, dists = loss_dist(accelerator, base_model, dataset.unshuffled_train_loader, center_model, centers)
    losses, dists = loss_dist_new_center(accelerator, base_model, dataset.unshuffled_train_loader, center_model, centers)
    
    sorted_indices = [index for index, value in sorted(enumerate(scores), key=lambda pair: pair[1])]
    
    sorted_losses = [losses[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_dists = [dists[i] for i in sorted_indices]
    
    num_intervals = 20
    interval = len(scores) // num_intervals
    inter_losses = []
    inter_scores = []
    inter_dists = []
    indexes = []
    
    for i in range(num_intervals):
        start, end = i * interval, (i+1) * interval
        indexes.append(start)
        inter_losses.append(avg(sorted_losses[start:end]))
        inter_scores.append(avg(sorted_scores[start:end]))
        inter_dists.append(avg(sorted_dists[start:end]))
    print(inter_losses, inter_scores, inter_dists)
    plt.figure()
    plt.plot(indexes, inter_losses, label='losses', color='red')
    plt.title('Tail Analysis')
    plt.xlabel('Number of Data')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Tail Loss.png')
    
    plt.figure()
    plt.plot(indexes, inter_scores, label='scores', color='green')
    plt.title('Tail Analysis')
    plt.xlabel('Number of Data')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('Tail Scores.png')
    
    plt.figure()
    plt.plot(indexes, inter_dists, label='distance', color='orange')
    plt.title('Tail Analysis')
    plt.xlabel('Number of Data')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig('Tail Distance1.png')
    


if __name__ == "__main__":
    main()