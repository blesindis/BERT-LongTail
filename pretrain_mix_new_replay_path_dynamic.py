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

# Local imports
import base_models
from utils import *
from Dataset import MixedData, ACLForLM

# replay
REPLAY = True
DATA_PER_PATH = 10
STEP = 50

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1102-mixed-stage1"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CENTER_FILE = 'centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
REPLAY_FILE = 'replay_dynamic_100.pth'
REPLAY_PATH = os.path.join(LOAD_PATH, REPLAY_FILE)
CONFIG_PATH = 'config/bert.json'

STORE_FOLDER = '1114-mixed-stage2-replay-path-dynamic-step50-10*(1102-mixed-stage1)'
STORE_PATH = os.path.join('outputs', STORE_FOLDER)

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0


def validate(model, val_loader, accelerator):
    losses = []
    print(len(val_loader))
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    loss = torch.mean(losses)
    
    return loss


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    layer_data = torch.tensor(np.array(layer_data)).to('cuda')
    return layer_data
        


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    dataset_old = MixedData(config, TRAIN_LEN, VAL_LEN)

    centers = load_layer_data(CENTER_PATH)
    
    
    if REPLAY:
        replay_alternatives = torch.load(REPLAY_PATH, map_location='cuda') # {path: {input_ids: , labels: }}
           
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader_old = dataset_old.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old)
    
    # train
    replay_rate_epoch = 0
    replay_effective = {}
    step = 20
    for epoch in range(num_epochs):
        model.train()
                
        losses = []
        replay_rate_batch = 0
        for i, batch in enumerate(train_loader):      
            # forward layer by layer to get center path
            h_ = model.bert.embeddings(batch['input_ids'])
            batch_cluster_labels = None
            
            for j in range(config.num_hidden_layers):
                cluster_list = model.bert.layers.layers[j].routing(h_)  
                cluster_labeles = get_cluster_labels(cluster_list)
                
                h_ = model.bert.layers.layers[j](h_, batch['attention_mask'])
                
                if j == 0:
                    batch_cluster_labels = cluster_labeles  
                else:
                    batch_cluster_labels = torch.cat((batch_cluster_labels, cluster_labeles), dim=1)
            
            scores = model.head(h_)
            loss = model.criterion(scores.view(-1, config.vocab_size), batch['labels'].view(-1))
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                        
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            # lr_scheduler.step()  
                        
            # replay layerwise by path
            loss = 0
            if REPLAY:
                if i % STEP == 0:
                    replay_batch = None
                    for l in range(batch_cluster_labels.shape[0]):
                        path = tuple(batch_cluster_labels[l].numpy())
                        if path in replay_alternatives:
                            replay_effective[path] = 1
                            input_ids = replay_alternatives[path]['input_ids'][:DATA_PER_PATH]
                            labels = replay_alternatives[path]['labels'][:DATA_PER_PATH]
                                
                            if replay_batch:
                                replay_batch['input_ids'] = torch.cat((replay_batch['input_ids'], input_ids.to('cuda')), dim=0)
                                replay_batch['labels'] = torch.cat((replay_batch['labels'], labels.to('cuda')), dim=0)
                            else:
                                replay_batch = {'input_ids': input_ids, 'labels': labels}                            
                    if replay_batch:
                        replay_batch['attention_mask'] = torch.ones(replay_batch['input_ids'].shape[:2]).to('cuda')
                        # replay_batch = {key: tensor.to('cuda') for key, tensor in replay_batch.items()}
                        l = 0         
                        # loss = 0           
                        while l < replay_batch['input_ids'].shape[0]:                    
                            batch_cur = {'input_ids': replay_batch['input_ids'][l:l+config.batch_size], 'attention_mask': replay_batch['attention_mask'][l:l+config.batch_size], 'labels': replay_batch['labels'][l:l+config.batch_size]}
                            loss_cur, _ = model(**batch_cur)
                            # loss += loss_cur
                            optimizer.zero_grad()
                            loss_cur.backward()
                            optimizer.step()
                            l += config.batch_size
                        replay_rate_batch += replay_batch['input_ids'].shape[0] / batch_cluster_labels.shape[0]                    
                            
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
            lr_scheduler.step() 
            
            # with torch.no_grad():
            #     replay_loss, _ = model(**replay_batch)
            #     print(f'replay loss: {replay_loss}')
                    
        if REPLAY:                            
            replay_rate_batch /= (len(train_loader) / STEP)
            replay_rate_epoch += replay_rate_batch          

        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        loss_valid_old = validate(model, val_loader_old, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Valid Loss Old: {loss_valid_old}')
        
        if REPLAY:
            writer.add_scalar('replay_rate', replay_rate_batch, epoch)
            writer.add_scalar('replay_effective', len(replay_effective) / len(replay_alternatives), epoch)
        writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar('perplexity_valid', loss_valid, epoch)
        writer.add_scalar('perplexity_valid_old', loss_valid_old, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        step += 1
        
    if REPLAY:
        print("Replay rate: ", replay_rate_epoch / num_epochs)
        print("Replay path effective rate: ", len(replay_effective) / len(replay_alternatives))
    
    accelerator.save_state(STORE_PATH)
    if REPLAY:
        torch.save(replay_effective, os.path.join(STORE_PATH, 'replay_effective_2.pth'))
    

if __name__ == '__main__':
    main()