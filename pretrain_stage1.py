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


# Constants and Configurations
SEED = 45


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
    parser.add_argument('--load_path', type=str, default="./output-0-saveall-1016", help='Path to load model parameters.')
    parser.add_argument('--config_path', type=str, default='config/bert.json', help='Path to BERT config file.')
    parser.add_argument('--train_on_newdata', action='store_true', help='Flag to train on new data.')
    parser.add_argument('--replay_layerwise', action='store_true', help='Flag to replay layer-wise.')
    parser.add_argument('--replay_decoder', action='store_true', help='Flag to replay in the decoder layer.')
    
    return parser.parse_args()


def validate(model, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):        
        with torch.no_grad():
            batch.to('cuda')
            loss, loss_dict, layer_outputs = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    # perplexity = torch.exp(perplexity)
    
    return perplexity


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cpu')
    layer_data = list(layer_data_dict.values())
    return layer_data


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
    train_on_newdata,
    replay_layerwise,
    replay_decoder
):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    pre_test_loader = dataset_pre.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter("log/" + 'bert')
    
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    accelerator.load_state(load_path)
    
    if replay_layerwise:
        layer_outputs = load_layer_data(os.path.join(load_path, 'layer_outputs.pth'))
        layer_outputs = [data.requires_grad_(True) for data in layer_outputs]
        layer_inputs = load_layer_data(os.path.join(load_path, 'layer_inputs.pth'))
        layer_labels = load_layer_data(os.path.join(load_path, 'layer_labels.pth'))
        layer_attns = load_layer_data(os.path.join(load_path, 'layer_attns.pth'))
        print(len(layer_outputs), len(layer_attns))
    
    if replay_decoder:
        decoder_outputs = load_layer_data(os.path.join(load_path, 'decoder_outputs.pth'))
    
    model.to('cuda')
    
    # freeze decoder
    # for param in model.head.parameters():
    #     param.requires_grad = False
        

    for epoch in range(num_epochs):
        model.train()
        """train origin bert (MLM only)"""
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        losses = []
        for i, batch in enumerate(train_loader):   
            # train on new data
            if train_on_newdata:
                batch = batch_to_cuda(batch)
                
                loss, _, _ = model(**batch)
                losses.append(accelerator.gather(loss.repeat(batch_size)))
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()    
                
                batch = batch_to_cpu(batch)
                torch.cuda.empty_cache()
            
            # replay in former 12 layers
            if replay_layerwise:
                for l in range(12):      
                    # update layer              
                    batch_layer = {'hidden_states': layer_inputs[l], 'attention_mask': layer_attns[l]}
                    batch_layer = batch_to_cuda(batch_layer)
                    
                    output = model.bert.layers.layers[l](**batch_layer)
                    
                    batch_layer = batch_to_cpu(batch_layer)
                    
                    output_std = layer_outputs[l].to('cuda')
                    layer_loss = mse_loss(output, output_std)
                    
                    local_optimizer = optim.AdamW(model.bert.layers.layers[l].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)             
                    local_optimizer.zero_grad()
                    layer_loss.backward(retain_graph=True)
                    local_optimizer.step()
                    
                    # update decoder
                    for k in range(l+1, 12):
                        batch_ = {'hidden_states': output, 'attention_mask': layer_attns[l]}
                        batch_ = batch_to_cuda(batch_)
                        
                        output = model.bert.layers.layers[k](**batch_)
                        
                        batch_ = batch_to_cpu(batch_)
                        
                    scores = model.head(output)
                    
                    label = layer_labels[l].to('cuda')
                    mlm_loss = ce_loss(scores.view(-1, vocab_size), label.view(-1))
                    
                    label = label.to('cpu')
                    
                    # Disable gradient computation for all parameters
                    for param in model.parameters():
                        param.requires_grad = False

                    # Enable gradient computation only for model.head parameters
                    for param in model.head.parameters():
                        param.requires_grad = True
                    
                    mlm_optimizer = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
                    mlm_optimizer.zero_grad()
                    mlm_loss.backward(retain_graph=True)
                    mlm_optimizer.step()
                    
                    for param in model.parameters():
                        param.requires_grad = True
                    
                    
                    
            # replay in the decoder layer
            if replay_decoder:          
                mse_loss = nn.MSELoss()
                batch_old = {'input_ids': layer_inputs[epoch], 'attention_mask': layer_attns[epoch], 'labels': layer_labels[epoch]}
                
                batch_old = batch_to_cuda(batch_old)
                _, scores, _ = model(**batch_old)
                scores = scores.to('cuda')
                batch_old = batch_to_cpu(batch_old)
                
                decoder_outputs[epoch] = decoder_outputs[epoch].to('cuda')
                decoder_loss = mse_loss(scores, decoder_outputs[epoch])
                scores = scores.to('cpu')
                decoder_outputs[epoch] = decoder_outputs[epoch].to('cpu')
                local_optimizer = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)             
                local_optimizer.zero_grad()
                decoder_loss.backward()
                local_optimizer.step()

                
                
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        loss_test = validate(model, pre_test_loader, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, pre_Test Loss: {loss_test}')

        if accelerator.is_local_main_process:
            # writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('perplexity_test', loss_test, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state('./output-formal-2')
    

def main():
    set_seed(SEED)
    
    # Parse arguments
    args = parse_arguments()
    
    config = BertConfig.from_json_file(args.config_path)
    dataset = ACLForLM_small(config=config)
    dataset_pre = RestaurantForLM_small(config=config)
    
    model = base_models.BertWithSavers(config=config)
    model.to('cuda')
    
    train(
        model=model, 
        num_epochs=args.num_epochs, 
        vocab_size = config.vocab_size,
        batch_size = config.batch_size,
        dataset=dataset, 
        dataset_pre=dataset_pre, 
        load_path=args.load_path, 
        train_on_newdata=args.train_on_newdata, 
        replay_layerwise=args.replay_layerwise, 
        replay_decoder=args.replay_decoder
    )


if __name__ == "__main__":
    main()