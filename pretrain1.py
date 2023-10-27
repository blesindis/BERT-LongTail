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
from Dataset import ACLForLM_small, RestaurantForLM_small, Wikitext


SEED = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Training script for BERT model.")
    
    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--load_path', type=str, default="./output-0-saveall-1020", help='Path to load model parameters.')
    parser.add_argument('--store_path', type=str, default='moe-stage0', help='Path to store output model.')
    parser.add_argument('--config_path', type=str, default='config/bert.json', help='Path to BERT config file.')
    parser.add_argument('--is_stage0', action='store_true', help='Flag to indicate if it is the first time pretrain.')
    parser.add_argument('--initialize', action='store_true', help='Flag to initialize a BERT model.')
    # parser.add_argument('--train_on_newdata', action='store_true', help='Flag to train on new data.')
    # parser.add_argument('--replay_layerwise', action='store_true', help='Flag to replay layer-wise.')
    # parser.add_argument('--replay_decoder', action='store_true', help='Flag to replay in the decoder layer.')
    
    return parser.parse_args()


def validate(model, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):        
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def batch_to_cuda(batch):
    batch = {key: tensor.to('cuda') for key, tensor in batch.items()}
    return batch


def batch_to_cpu(batch):
    batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
    return batch


def train(
    # model/data params
    model, 
    num_epochs, 
    vocab_size,
    batch_size,
    dataset, 
    dataset_pre,
    load_path,
    store_path,
    is_stage0,
    initialize
):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    pre_test_loader = dataset_pre.val_loader
    num_updates = num_epochs * len(train_loader)
    
    # checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
    # model.load_state_dict(checkpoint)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', store_path))
    
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)

    for epoch in range(num_epochs):
        model.train()
        
        # initialize
        if initialize:            
            # use the first batch to initialize
            losses = []
            for i, batch in enumerate(train_loader):
                loss, _= model(**batch)
                losses.append(accelerator.gather(loss.repeat(batch_size)))
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()    
                break
                          
            loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            loss_valid = validate(model, val_loader, accelerator)
            accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')

            writer.add_scalar('initialize: perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('initialize: perplexity_valid', loss_valid, epoch)
            writer.add_scalar('initialize: learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
        # stage 0
        elif is_stage0:
            # train
            losses = []
            for i, batch in enumerate(train_loader):
                loss, _= model(**batch)
                losses.append(accelerator.gather(loss.repeat(batch_size)))
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()    
                          
            loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            loss_valid = validate(model, val_loader, accelerator)
            accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')

            writer.add_scalar('stage0: perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('stage0: perplexity_valid', loss_valid, epoch)
            writer.add_scalar('stage0: learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(os.path.join('outputs', store_path))
    

def main():
    set_seed(SEED)
    
    args = parse_arguments()
    
    config = BertConfig.from_json_file(args.config_path)
    
    if args.initialize:
        model = base_models.BertForMLM(config)
        dataset = RestaurantForLM_small(config=config)
    
    dataset_pre = RestaurantForLM_small(config=config)
    
    train(
        model=model, 
        num_epochs=args.num_epochs, 
        vocab_size = config.vocab_size,
        batch_size = config.batch_size,
        dataset=dataset, 
        dataset_pre=dataset_pre, 
        load_path=args.load_path, 
        store_path=args.store_path,
        is_stage0=args.is_stage0,
        initialize=args.initialize
    )


if __name__ == "__main__":
    main()