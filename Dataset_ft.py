from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
import torchvision
import multiprocessing
from transformers import BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, load_from_disk
import os
from collections import Counter
from itertools import chain
import torch
import numpy as np
import sys
import json


class SST2:
    def preprocess(self, config, path):
        # Load dataset from TSV file
        train_file = '/home/mychen/ER_TextSpeech/BERT/data/datasets/SST-2/train.tsv'
        val_file = '/home/mychen/ER_TextSpeech/BERT/data/datasets/SST-2/dev.tsv'
        
        # dataset_train = load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['sentence', 'label'])['train']
        # dataset_val = load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['sentence', 'label'])['train']
        datasets = DatasetDict({
            'train': load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['sentence', 'label'])['train'],
            'validation': load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['sentence', 'label'])['train'],
        })
        
        # column_names = datasets["train"].column_names
        column_names = ['sentence']
        # print(column_names)
        # Assuming the file has 'text' and 'label' columns
        def tokenize_function(examples):
            # Tokenize the texts
            return self.tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=self.block_size)

        # Tokenize the dataset
        # tokenized_train = dataset_train.map(tokenize_function, batched=True, remove_columns=column_names)
        # tokenized_val = dataset_val.map(tokenize_function, batched=True, remove_columns=column_names)
        # tokenized_datasets = DatasetDict({'train': tokenized_train, 'validation': tokenized_val})
        # tokenized_train.save_to_disk(path)
        tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=column_names)
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')


        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/SST-2", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
            
        lm_datasets = load_from_disk(path)
        # train_data = Subset(lm_datasets['train'], range(train_len))
        # val_data = Subset(lm_datasets['validation'], range(val_len))
        train_data = lm_datasets['train']
        val_data = lm_datasets['validation']
        
        # Create PyTorch DataLoader
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        def collate_fn(batch):
            # print(batch)
            # texts = [item[0] for item in batch]  # Assuming each item in the batch is a tuple (text, label)
            # print([item for item in batch])
            labels = torch.tensor([int(item['label']) for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            # sentences = [item['sentence'] for item in batch]
            # batch_new = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
            # batch_new = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            # batch_new['labels'] = labels
            batch_new = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
            return batch_new
        # Create a data collator that dynamically pads the batches so that each batch has the same length
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)