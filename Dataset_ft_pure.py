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


class SST2_pure:
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    
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
        column_names = datasets['train'].column_names
        # print(column_names)
        # Assuming the file has 'text' and 'label' columns
        def tokenize_function(examples):
            # Tokenize the texts
            return self.tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=self.block_size, return_special_tokens_mask=True)

        # Tokenize the dataset
        # tokenized_train = dataset_train.map(tokenize_function, batched=True, remove_columns=column_names)
        # tokenized_val = dataset_val.map(tokenize_function, batched=True, remove_columns=column_names)
        # tokenized_datasets = DatasetDict({'train': tokenized_train, 'validation': tokenized_val})
        # tokenized_train.save_to_disk(path)
        tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=column_names)
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {self.block_size}",
        )
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')


        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets_pure_ft/tokenized/SST-2", str(self.block_size))
        if not config.ft_preprocessed:
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

        # Create a data collator that dynamically pads the batches so that each batch has the same length
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        
class GAD:
    def preprocess(self, config, path):
        # Load dataset from TSV file
        train_file = '/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/BIO/TASK/REdata/GAD/1/train.tsv'
        val_file = '/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/BIO/TASK/REdata/GAD/1/test.tsv'
        
        dataset_train = load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['sentence', 'label'])['train']
        dataset_val = load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['index', 'sentence', 'label'])['train']
        # datasets = DatasetDict({
        #     'train': load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['sentence', 'label'])['train'],
        #     'validation': load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['index', 'sentence', 'label'])['train'],
        # })
        
        # column_names = datasets["train"].column_names
        # column_names = ['sentence', 'index']
        train_column_names = ['sentence']
        val_column_names = ['sentence', 'index']
        # print(column_names)
        # Assuming the file has 'text' and 'label' columns
        def tokenize_function(examples):
            # Tokenize the texts
            return self.tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=self.block_size)

        # Tokenize the dataset
        tokenized_train = dataset_train.map(tokenize_function, batched=True, remove_columns=train_column_names)
        tokenized_val = dataset_val.map(tokenize_function, batched=True, remove_columns=val_column_names)
        tokenized_datasets = DatasetDict({'train': tokenized_train, 'validation': tokenized_val})
        tokenized_datasets.save_to_disk(path)
        # tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=column_names)
        # tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')


        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/tokenized/BIO/TASK/REdata/GAD/1", str(self.block_size))
        if not config.ft_preprocessed:
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
        
        
# class Overruling:
#     def preprocess(self, config, path):
#         # Load dataset from TSV file
#         train_file = '/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/LEGAL/train.csv'
#         val_file = '/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/LEGAL/dev.csv'
        

#         # dataset_train = load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['sentence', 'label'])['train']
#         # dataset_val = load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['index', 'sentence', 'label'])['train']
#         datasets = DatasetDict({
#             'train': load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['label', 'sentence'])['train'],
#             'validation': load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['label', 'sentence'])['train'],
#         })
        
#         # column_names = datasets["train"].column_names
#         column_names = ['sentence']
#         # train_column_names = ['sentence']
#         # val_column_names = ['sentence', 'index']
#         # print(column_names)
#         # Assuming the file has 'text' and 'label' columns
#         def tokenize_function(examples):
#             # Tokenize the texts
#             return self.tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=self.block_size)

#         # Tokenize the dataset
#         # tokenized_train = dataset_train.map(tokenize_function, batched=True, remove_columns=train_column_names)
#         # tokenized_val = dataset_val.map(tokenize_function, batched=True, remove_columns=val_column_names)
#         # tokenized_datasets = DatasetDict({'train': tokenized_train, 'validation': tokenized_val})
#         # tokenized_datasets.save_to_disk(path)
#         tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=column_names)
#         tokenized_datasets.save_to_disk(path)

#         return tokenized_datasets

#     def __init__(self, config):
#         self.block_size = config.seq_len
#         self.batch_size = config.batch_size
#         self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')


#         path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/tokenized/LEGAL", str(self.block_size))
#         if not config.ft_preprocessed:
#             self.preprocess(config, path)
            
#         lm_datasets = load_from_disk(path)
#         # train_data = Subset(lm_datasets['train'], range(train_len))
#         # val_data = Subset(lm_datasets['validation'], range(val_len))
#         train_data = lm_datasets['train']
#         val_data = lm_datasets['validation']
        
#         # Create PyTorch DataLoader
#         seed = 42
#         torch.manual_seed(seed)
#         np.random.seed(seed)

#         def collate_fn(batch):
#             # print(batch)
#             # texts = [item[0] for item in batch]  # Assuming each item in the batch is a tuple (text, label)
#             # print([item for item in batch])
#             labels = torch.tensor([int(item['label']) for item in batch])
#             input_ids = torch.tensor([item['input_ids'] for item in batch])
#             attention_mask = torch.tensor([item['attention_mask'] for item in batch])
#             # sentences = [item['sentence'] for item in batch]
#             # batch_new = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
#             # batch_new = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
#             # batch_new['labels'] = labels
#             batch_new = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
#             return batch_new
#         # Create a data collator that dynamically pads the batches so that each batch has the same length
#         data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
#         self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
#         # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
#         self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        
class Overruling():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        data_files = {'train':'/home/mychen/ER_TextSpeech/BERT/data/ft_datasets/LEGAL/overruling.csv'}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split=f"train[:{33}%]"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split=f"train[{33}%:]",
        )
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence1'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence1"])
        tokenized_datasets.set_format("torch")
        
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        # print(tokenized_datasets['train'][0])

        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)