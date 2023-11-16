import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import BertConfig

# Local imports
import base_models
from utils import *
from Dataset import MixedData, ACLForLM

# cluster config
SAMPLE_BATCHES = 100

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
REPLAY_FILE = 'replay_dynamic_100.pth'
REPLAY_PATH = os.path.join(STORE_PATH, REPLAY_FILE)
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

    """generate replay alternatives for dynamic replay"""
    path_dict = {}
    
    data_cluster_labels = None
    with torch.no_grad():
        for i, batch in enumerate(train_loader):         
            if i >= SAMPLE_BATCHES: 
                break
            
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
                    
            for l in range(batch_cluster_labels.shape[0]):
                path = tuple(batch_cluster_labels[l].numpy())
                if path in path_dict:
                    path_dict[path]['input_ids'] = torch.cat((path_dict[path]['input_ids'], batch['input_ids'][l].unsqueeze(0)), dim=0)
                    path_dict[path]['labels'] = torch.cat((path_dict[path]['labels'], batch['labels'][l].unsqueeze(0)), dim=0)
                else:
                    path_dict[path] = {'input_ids': batch['input_ids'][l].unsqueeze(0), 'labels': batch['labels'][l].unsqueeze(0)}
                    
            
            if i == 0:
                data_cluster_labels = batch_cluster_labels
            else:
                data_cluster_labels = torch.cat((data_cluster_labels, batch_cluster_labels), dim=0)
            batch = {key: tensor.to('cpu') for key, tensor in batch.items()}

    print("Number of total paths: ", len(path_dict))    
    torch.save(path_dict, REPLAY_PATH)    
    
    # """Select a batch for single batch replay"""
    # max_path = None
    # max_len = 0
    # for path, data in path_dict.items():
    #     if data['input_ids'].shape[0] > max_len:
    #         max_len = data['input_ids'].shape[0]
    #         max_path = data
    
    # data_outputs = []
    # with torch.no_grad():
    #     for i in range(max_path['input_ids'].shape[0]):
    #         input_ids = max_path['input_ids'][i].unsqueeze(0).to('cuda')
    #         output = model.bert(input_ids, torch.ones(input_ids.shape[:2]).to('cuda'))                                        
    #         data_outputs.append(output.mean(axis=1))
    # data_outputs = torch.cat(data_outputs, dim=0)
    # sample_indexes = sample_by_cluster(data_outputs.cpu(), k=8, n_samples=8)
    # single_batch = {'input_ids': max_path['input_ids'][sample_indexes], 'labels': max_path['labels'][sample_indexes], 'attention_mask': torch.ones(64, 128)}
    # torch.save(single_batch, REPLAY_PATH)
    
    """Go through one forward for new training data to select replay data for static replay"""
    # replay_batches = {}
    # replay_effective = {}
    # average_replay_rate = 0
    # with torch.no_grad():
    #     for i, batch in enumerate(new_train_loader):             
            
    #         h_ = model.bert.embeddings(batch['input_ids'])
    #         batch_cluster_labels = None
            
    #         for j in range(config.num_hidden_layers):
    #             cluster_list = model.bert.layers.layers[j].routing(h_)  
    #             cluster_labeles = get_cluster_labels(cluster_list)
                
    #             h_ = model.bert.layers.layers[j](h_, batch['attention_mask'])
                
    #             if j == 0:
    #                 batch_cluster_labels = cluster_labeles  
    #             else:
    #                 batch_cluster_labels = torch.cat((batch_cluster_labels, cluster_labeles), dim=1)
            
    #         replay_batch = None        
    #         for l in range(batch_cluster_labels.shape[0]):
    #             path = tuple(batch_cluster_labels[l].numpy())
    #             if path in path_dict:
    #                 replay_effective[path] = 1
    #                 if replay_batch:
    #                     replay_batch['input_ids'] = torch.cat((replay_batch['input_ids'], path_dict[path]['input_ids'][0].unsqueeze(0)), dim=0)
    #                     replay_batch['labels'] = torch.cat((replay_batch['labels'], path_dict[path]['labels'][0].unsqueeze(0)), dim=0)
    #                 else:
    #                     replay_batch = {'input_ids': path_dict[path]['input_ids'][0].unsqueeze(0), 'labels': path_dict[path]['labels'][0].unsqueeze(0)}
    #         if replay_batch:
    #             replay_batch['attention_mask'] = torch.ones(replay_batch['input_ids'].shape[:2])
    #             average_replay_rate += replay_batch['input_ids'].shape[0] / batch_cluster_labels.shape[0]    
                  
    #         replay_batches[i] = replay_batch
    #         batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
    
    # print(f'Number of effective replay path: {len(replay_effective)}')
    # print(f'Average replay rate: {average_replay_rate / len(new_train_loader)}')
    # torch.save(replay_batches, REPLAY_PATH)
    
    """pic path distribution"""
    # path_len = []
    # for value in path_dict.values():
    #     path_len.append(len(value['input_ids']))
    #     # print(len(value['input_ids']))
    
    # plt.figure()
    # plt.plot(np.arange(len(path_len)), path_len)
    # plt.xlabel('Path Index')
    # plt.ylabel('Number of Data on Path')
    # plt.title('Path Distribution')
    # plt.grid(True)
    # plt.savefig('path distribution.png')
    
if __name__ == '__main__':
    main()