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
from Dataset import MixedData, ACLForLM


DATASET_SIZE = 10000
REPLAY = False


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Training script for BERT model.")
    
    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--load_path', type=str, default="1027-mixed", help='Path to load model parameters.')
    parser.add_argument('--store_path', type=str, default="1031-mixed-new-noreplay", help='Path to store model parameters.')
    parser.add_argument('--config_path', type=str, default='config/bert.json', help='Path to BERT config file.')
    
    return parser.parse_args()


def validate(model, val_loader, accelerator):
    losses = []
    print(len(val_loader))
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    # perplexity = torch.exp(perplexity)
    
    return perplexity


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    layer_data = torch.tensor(np.array(layer_data))
    return layer_data


def copy_parameters(source_module, target_module):
    for source_param, target_param in zip(source_module.parameters(), target_module.parameters()):
        target_param.data.copy_(source_param.data)


def main():
    args = parse_arguments()
    
    num_epochs = args.num_epochs
    config_path = args.config_path
    load_path = args.load_path
    store_path = args.store_path
    
    config = BertConfig.from_json_file(config_path)
    
    dataset = ACLForLM(config, dataset_len=DATASET_SIZE)
    dataset_old = MixedData(config, dataset_len=DATASET_SIZE)
    
    # new center
    centers = load_layer_data(os.path.join('outputs', load_path, 'new_centers.pth'))
    centers = torch.squeeze(centers, 2)
    
    # old center
    # centers = load_layer_data(os.path.join('outputs', '1027-mixed-warmup', 'centers.pth'))
    
    if REPLAY:
        layer_data_dict = torch.load(os.path.join('outputs', load_path, 'sample_inputs.pth'), map_location='cuda')
        sample_inputs = list(layer_data_dict.values())
        layer_data_dict = torch.load(os.path.join('outputs', load_path, 'sample_outputs.pth'), map_location='cuda')
        sample_outputs = list(layer_data_dict.values())
    
    model = base_models.BertWithMOE(config, centers=centers)
    checkpoint = torch.load(os.path.join('outputs', load_path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader_old = dataset_old.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0., betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', store_path))
    
    model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old)
    
    loss_valid_old = validate(model, val_loader_old, accelerator) 
    print(f'Valid Loss After Updating Centers: {loss_valid_old}')
    
    # train
    for epoch in range(num_epochs):
        model.train()
        
        mse_loss = nn.MSELoss()
        
        
        # train new data
        losses = []
        for i, batch in enumerate(train_loader):            
            loss, _ = model(**batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()  
            
            # replay layerwise by cluster
            if REPLAY:
                for j in range(config.num_hidden_layers):
                    for k in range(config.num_experts):
                        inputs = sample_inputs[j][k]
                        outputs = model.bert.layers.layers[j].experts[k](inputs, torch.ones(inputs.shape[0], inputs.shape[1]).to('cuda'))
                        outputs_std = sample_outputs[j][k]
                        
                        layer_loss = mse_loss(outputs_std, outputs)
                        local_optimizer = optim.AdamW(model.bert.layers.layers[j].experts[k].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)             
                        local_optimizer.zero_grad()
                        layer_loss.backward(retain_graph=True)
                        local_optimizer.step()
                        
            
        # train old data

        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        loss_valid_old = validate(model, val_loader_old, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Valid Loss Old: {loss_valid_old}')
        

        writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar('perplexity_valid', loss_valid, epoch)
        writer.add_scalar('perplexity_valid_old', loss_valid_old, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(os.path.join('outputs', store_path))
    

if __name__ == '__main__':
    main()