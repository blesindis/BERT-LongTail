"""
With the architeture of cluster layerwise moe, check the CF caused by new data on old data of different paths,
the path difference mainly lies in the number of overlapping clusters on the path
"""
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import BertConfig

# Local imports
import base_models
from utils.sample_utils import *
from Dataset import MixedData, ACLForLM

# cluster config
SAMPLE_BATCHES = 10

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1027-mixed-warmup"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = '1102-mixed-stage1'
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CENTER_FILE = 'centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
CONFIG_PATH = 'config/bert.json'

# training config
lr = 1e-3
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


def validate_batch(model, inputs, labels):
    losses = []
    for i in range(len(inputs)):                
        with torch.no_grad():
            loss, _ = model(inputs[i].unsqueeze(0), torch.ones(1, inputs.shape[1]).to('cuda'), labels[i].unsqueeze(0))
            losses.append(loss)
                
    # loss = torch.mean(loss)
    return torch.tensor(losses) 


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())    
    layer_data = torch.tensor(np.array(layer_data)).to('cuda')
    return layer_data


def get_cluster_labels(cluster_list):
    # return a tensor, where each element indicates the cluster of data with current index
    num_data = sum(len(sublist) for sublist in cluster_list)
    cluster_labels = torch.zeros((num_data, 1))
    
    for i, sublist in enumerate(cluster_list):
        for element in sublist:
            cluster_labels[element] = i
    
    return cluster_labels


def main():  
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = MixedData(config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    new_dataset = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    centers = load_layer_data(CENTER_PATH)
    
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    new_train_loader = new_dataset.train_loader
    
    accelerator = Accelerator()
    
    model, train_loader, val_loader, new_train_loader = accelerator.prepare(model, train_loader, val_loader, new_train_loader)
    
    test_data_inputs = []
    test_data_labels = []
    test_data_outputs = []

    data_cluster_labels = None
    with torch.no_grad():
        for i, batch in enumerate(new_train_loader):         
            if i >= SAMPLE_BATCHES: 
                break
            
            h_ = model.bert.embeddings(batch['input_ids'])
            batch_cluster_labels = None
            
            for j in range(config.num_hidden_layers):
                # get routing result
                cluster_list = model.bert.layers.layers[j].routing(h_)  
                # get layer output
                h_ = model.bert.layers.layers[j](h_, batch['attention_mask'])
                if j == 0:
                    batch_cluster_labels = get_cluster_labels(cluster_list)   
                else:
                    batch_cluster_labels = torch.cat((batch_cluster_labels, get_cluster_labels(cluster_list)), dim=1)
                    if j == config.num_hidden_layers - 1:                        
                        for k in range(config.num_experts):
                            if i == 0:
                                test_data_inputs.append(batch['input_ids'][batch_cluster_labels[:,1] == k])
                                test_data_labels.append(batch['labels'][batch_cluster_labels[:,1] == k])
                                test_data_outputs.append(h_[[batch_cluster_labels[:,1] == k]])
                            else:
                                test_data_inputs[k] = torch.cat((test_data_inputs[k], batch['input_ids'][batch_cluster_labels[:,1] == k]), dim=0)
                                
                                test_data_labels[k] = torch.cat((test_data_labels[k], batch['labels'][batch_cluster_labels[:,1] == k]), dim=0)
                                test_data_outputs[k] = torch.cat((test_data_outputs[k], h_[batch_cluster_labels[:,1] == k]), dim=0)
                                
                    
                
            if i == 0:
                data_cluster_labels = batch_cluster_labels
            else:
                data_cluster_labels = torch.cat((data_cluster_labels, batch_cluster_labels), dim=0)
            batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
            
    for d in test_data_inputs:
        print(len(d))
    
    """Test the impact of new data on old data of different paths"""
    dists = []
    loss_change = []
    for i, batch in enumerate(new_train_loader):
        
        pre_losses = []
        for k in range(4):
            pre_loss = validate_batch(model, test_data_inputs[k][:64], test_data_labels[k][:64])
            pre_losses.append(pre_loss)
            
        model.train()
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
        accelerator = Accelerator()
        model, optimizer = accelerator.prepare(model, optimizer)
        
        output = model.bert(batch['input_ids'], batch['attention_mask'])
        scores = model.head(output)
        loss = model.criterion(scores.view(-1, config.vocab_size), batch['labels'].view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        
        for k in range(4):
            post_loss = validate_batch(model, test_data_inputs[k][:64], test_data_labels[k][:64])              
            loss_change.append(post_loss - pre_losses[k])
            dist = torch.cdist(output.view(1,-1).double(), test_data_outputs[k][:64].view(len(test_data_outputs[k][:64]), -1).double()).squeeze(0)
            print(dist)
            dists.append(dist)
            
        break
    
    # draw pic
    colors = plt.cm.jet(np.linspace(0, 1, config.num_experts))
    plt.figure(figsize=(10, 6))

    for i in range(config.num_experts):
        # print(len(dists[i]))
        # print(len(loss_change[i]))
        # plt.scatter(dists[i].cpu().detach( ).numpy(), loss_change[i].cpu().detach().numpy(), color=colors[i], label=f'Cluster {i+1}')
        dist = dists[i].cpu().detach().float().numpy()
        
        loss = loss_change[i].cpu().detach().numpy()
        print(dist.dtype, loss.dtype)
        print(loss.shape, dist.shape)
        if len(dist) == len(loss):
            plt.scatter(dist, loss, color=colors[i], label=f'Cluster {i+1}')
        else:
            print(f'Skipping Cluster {i+1} due to mismatched lengths.')

    plt.xlabel('Distance')
    plt.ylabel('Loss Change')
    plt.title('Distance vs Loss Change for Different Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('result1.png')
    # print(data_cluster_labels)
    
if __name__ == '__main__':
    main()