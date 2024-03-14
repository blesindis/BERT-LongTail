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
from transformer.FT_BERT import BertForSequenceClassification
from transformer.BERT import BertForMLM

# Local imports
from Dataset import Wikipedia ,BookCorpus, BERTPretrain
from Dataset_ft import SST2
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    copy_parameters
)

LOAD_CHECKPOINT = False

# train and validation size for pretrain
TRAIN_LEN = 250000
VAL_LEN = 1600

# folder paths
LOAD_FOLDER = "bert(128)-bs64-3epoch-lr1-bert"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = "bert(128)-bs64-3epoch-lr1-bert-FT**"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 10
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


def validate_acc(model, val_loader, accelerator):
    total_accuracy = 0
    total_examples = 0
    for batch in val_loader:
        # print(batch)     
        labels = batch['labels']
        with torch.no_grad():
            loss, logits = model(**batch)
        batch_accuracy = compute_accuracy(logits, labels)
        total_accuracy += batch_accuracy * labels.size(0)
        total_examples += labels.size(0)
        
    overall_accuracy = total_accuracy / total_examples
    return overall_accuracy


def main():
    set_seed(45)
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = SST2(config=config)
    
    # LOAD Pre-trained Model
    base_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    base_model.load_state_dict(checkpoint)
    
    # Incorporate into finetune model
    model = BertForSequenceClassification(config, num_labels=2)
    copy_parameters(base_model.bert, model.bert)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    print(len(train_loader), len(val_loader))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler, val_loader, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, val_loader, train_loader)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        
        losses = []        
        for i, batch in enumerate(train_loader):                 
            loss, _ = model(**batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
                
            if step % 100 == 0:
                loss_train = torch.mean(torch.cat(losses)[:6400])
                acc_valid = validate_acc(model, val_loader, accelerator)
                accelerator.print(f'iteration:{step} , Train Loss: {loss_train}, Valid Acc: {acc_valid}')

                writer.add_scalar(f'perplexity_train_epoch', loss_train, step)
                writer.add_scalar(f'accuracy_valid', acc_valid, step)
                writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
                
                losses = []   
            if step % 5000 == 0:
                accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
            step += 1
        
    accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    