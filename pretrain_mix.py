import os
import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from utils.train_utils import (
    validate,
    load_layer_data,
)
from Dataset import MixedData


# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1027-mixed-warmup"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = '1113-stage1-2expert'
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CENTER_FILE = '2expert-centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
CONFIG_PATH = 'config/bert.json'

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0


def copy_parameters(source_module, target_module):
    for source_param, target_param in zip(source_module.parameters(), target_module.parameters()):
        target_param.data.copy_(source_param.data)


def main():   
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = MixedData(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    centers = load_layer_data(CENTER_PATH)
    
    model = base_models.BertWithMOE(config, centers)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    """initialize model parameters by copy bert"""
    model_warmup = base_models.BertForMLM(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model_warmup.load_state_dict(checkpoint)
    # Common: Embedding & Decoder
    copy_parameters(model_warmup.bert.embeddings, model.bert.embeddings)
    copy_parameters(model_warmup.head, model.head)
        
    # Experts: 
    for i in range(config.num_hidden_layers):
        for j in range(config.num_experts):
            copy_parameters(model_warmup.bert.encoders.layers[i], model.bert.layers.layers[i].experts[j])
    
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)        
    
    
    # train
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

        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')                

        writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar('perplexity_valid', loss_valid, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(STORE_PATH)
    

if __name__ == '__main__':
    main()