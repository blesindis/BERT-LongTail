import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import RestaurantForLM_small, ACLForLM_small, LegalForLM_small
from utils import *


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
    parser.add_argument('--store_path', type=str, default='moe-stage0', help='Path to store output model.')
    parser.add_argument('--config_path', type=str, default='config/bert.json', help='Path to BERT config file.')

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
    

def main():
    set_seed(SEED)
    
    args = parse_arguments()
    
    config = BertConfig.from_json_file(args.config_path)
    datasetA = LegalForLM_small(config=config)
    datasetR = RestaurantForLM_small(config)
    
    num_epochs = args.num_epochs
    
    model = base_models.BertForMLM(config)
    checkpoint = torch.load(os.path.join('outputs/Restaurant half 1', 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loaderA = datasetA.train_loader
    train_loaderR = datasetR.train_loader
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, betas=[0.9, 0.999], eps=1e-6)
    # accelerator = Accelerator()
    
    # num_updates = num_epochs * len(train_loader)
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader)
    
    layer_outputs = []
    for i, batch in enumerate(train_loaderA):
        print('batch:', i)
        if i >= 20:
            break
        with torch.no_grad():            
            h_ = model.bert.embeddings(batch['input_ids'])
            for j in range(12):
                h_ = model.bert.encoders.layers[j](h_, batch['attention_mask'])
                if i == 0:
                    layer_outputs.append(h_)
                else:
                    layer_outputs[j] = torch.cat((layer_outputs[j], h_), dim=0)
                    
    layer_outputsR = []
    for i, batch in enumerate(train_loaderR):
        if i >= 20:
            break
        with torch.no_grad():
            h_ = model.bert.embeddings(batch['input_ids'])
            for j in range(12):
                h_ = model.bert.layers.layers[j](h_, batch['attention_mask'])
                if i == 0:
                    layer_outputsR.append(h_)
                else:
                    layer_outputsR[j] = torch.cat((layer_outputsR[j], h_), dim=0)
    
    for i, output in enumerate(layer_outputs):
        # output = torch.cat((output, layer_outputsR[i]), dim=0)
        data = output.mean(axis=1)
        data = data.cpu().numpy()
        
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        plt.figure(i)
        # plt.scatter(data_pca[:64*20,0], data_pca[:64*20,1], alpha=0.5, c='r')
        plt.scatter(data_pca[:,0], data_pca[:,1], alpha=0.5, c='b')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Result')
        plt.grid(True)
        plt.savefig('layer ' + str(i+1) + 'pca_result.png')
    


if __name__ == "__main__":
    main()