import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from transformer.FT_MTL import BertMTLForSequenceClassification
from transformer.FT_MTLLastTwo import BertMTLLastTwoForSequenceClassification
from transformer.MoMoTModelRouterMTLLastTwo import BertWithMoMoTModelRouterMTLLastTwo
from transformer.BERT import BertForMLM

# Local imports
from Dataset_ft import SST2, GAD, Overruling
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    copy_parameters, 
    load_layer_data_last
)
NUM_EXPERTS = 4
# folder paths
CENTER_MODEL_PATH = "outputs/wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, f'centers-{NUM_EXPERTS}-momoe-transformer-lastlayer.pth')
LOAD_FOLDER = "wiki(128)300w-bs64-1epoch-lr3-momot_model_router_mtl(lora-full, 4-4)_layer(last two)_router5000/checkpoint-46875"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = "ft-overruling-wiki(128)300w-bs64-1epoch-lr3-momot_model_router_mtl(lora-full, 4-4)_layer(last two)_router5000"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 20
lr = 3e-4
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


def validate_acc(model, val_loader, accelerator, center_model, centers):
    total_accuracy = 0
    total_examples = 0
    for batch in val_loader:
        # print(batch)     
        labels = batch['labels']
        with torch.no_grad():
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            _, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]   
            
            loss, logits = model(**batch, cluster_list=cluster_list)
        batch_accuracy = compute_accuracy(logits, labels)
        total_accuracy += batch_accuracy * labels.size(0)
        total_examples += labels.size(0)
        
    overall_accuracy = total_accuracy / total_examples
    return overall_accuracy


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config_path = os.path.join(LOAD_PATH, 'config.json')
    config = BertConfig.from_json_file(config_path)
    dataset = Overruling(config=config)
    
    center_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    center_model = accelerator.prepare(center_model)
    
    # LOAD Pre-trained Model
    centers = load_layer_data_last(CENTER_PATH)
    base_model = BertWithMoMoTModelRouterMTLLastTwo(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    base_model.load_state_dict(checkpoint)
    
    # Incorporate into finetune model
    model = BertMTLLastTwoForSequenceClassification(config, num_labels=2)
    copy_parameters(base_model.bert, model.bert)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    print(len(train_loader), len(val_loader))
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler, val_loader, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, val_loader, train_loader)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        
        losses = []        
        for i, batch in enumerate(train_loader):    
            with torch.no_grad():          
                cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
                hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
                
                h = hidden_states.mean(dim=1)
                
                dist = torch.cdist(h.double(), centers.double())
                _, min_indices = torch.min(dist, dim=1)
                cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]               
            loss, _ = model(**batch, cluster_list=cluster_list)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
                
            if step % 100 == 0:
                loss_train = torch.mean(torch.cat(losses)[:6400])
                acc_valid = validate_acc(model, val_loader, accelerator, center_model, centers)
                accelerator.print(f'iteration:{step} , Train Loss: {loss_train}, Valid Acc: {acc_valid}')

                writer.add_scalar(f'loss_train_epoch', loss_train, step)
                writer.add_scalar(f'accuracy_valid', acc_valid, step)
                writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
                
                losses = []   
            if step % 5000 == 0:
                accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
            step += 1
        
    accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    