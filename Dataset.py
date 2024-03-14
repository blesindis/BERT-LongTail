from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
import torchvision
import multiprocessing
from transformers import BertConfig, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, load_from_disk
import os
from collections import Counter
from itertools import chain
import torch
import numpy as np
import sys
import json


class Wikitext():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2

        raw_datasets = load_from_disk('/home/mychen/ER_TextSpeech/BERT/data/datasets/wikitext-2-raw')
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)
        lm_dataset.save_to_disk(path)
        return lm_dataset

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')

        path = os.path.join('/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/wikitext-2-raw', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        # data_collator = DistributedSampler(data_collator)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
          
class Wikitext103():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2

        raw_datasets = load_from_disk('/home/mychen/ER_TextSpeech/BERT/data/datasets/wikitext-103-raw')
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)
        lm_dataset.save_to_disk(path)
        return lm_dataset

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')

        path = os.path.join('/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/wikitext-103-raw', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        # data_collator = DistributedSampler(data_collator)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
    

class RestaurantForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/archen/yelp_restaurant.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{0.5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config, train_len, val_len):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/restaurant", str(self.block_size))
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/restaurant", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        train_data = Subset(lm_datasets['train'], range(train_len))
        val_data = Subset(lm_datasets['validation'], range(val_len))
        
        train_data = Subset(lm_datasets['train'], range(train_len))
        val_data = Subset(lm_datasets['validation'], range(val_len))
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class ACLForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/mychen/ER_TextSpeech/BERT/data/datasets/acl/acl_anthology.txt'}
        data_files = {'train': '/home/mychen/ER_TextSpeech/BERT/data/datasets/acl/acl_anthology.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{0.5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config, train_len, val_len):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        # self.batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/acl", str(self.block_size))
        # self.batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/acl", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        train_data = Subset(lm_datasets['train'], range(train_len))
        val_data = Subset(lm_datasets['validation'], range(val_len))
        
        train_data = Subset(lm_datasets['train'], range(train_len))
        val_data = Subset(lm_datasets['validation'], range(val_len))
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class PhoneForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/archen/phone.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        path = os.path.join("/home/archen/datasets/PhoneforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class CameraForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/archen/camera.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        path = os.path.join("/home/archen/datasets/CameraforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class PubMedForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/archen/format_pubmed_small.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        path = os.path.join("/home/archen/datasets/PubMedforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class MixedData():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path, input_path):
        data_files = {'train': input_path}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:5%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets
    def __init__(self, config, train_len, val_len):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        input_paths = [
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/acl/acl_anthology.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/legal/legal.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/restaurant/yelp_restaurant.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/review/review.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/phone/phone.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/camera/camera.txt'
        ]
        paths = [
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/acl',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/legal',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/restaurant',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/review',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/phone',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/camera'
        ]
        
        for i, path in enumerate(paths):
            paths[i] = os.path.join(path, str(self.block_size))
        
        if not config.preprocessed:
            for path, input_path in zip(paths, input_paths):
                self.preprocess(config, path, input_path)
        
        datasets = []
        for path in paths:
            datasets.append(load_from_disk(path))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        val_data = None
        train_data = None
        val_loader_set = []
        for dataset in datasets:
            if not val_data:
                val_data, train_data = dataset['validation'], dataset['train']
                # val_data = Subset(val_data, range(val_len))
                # train_data = Subset(train_data, range(train_len))
                val_loader_set.append(DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator))
            else:
                dataset_val, dataset_train = dataset['validation'], dataset['train']
                # dataset_val = Subset(dataset['validation'], range(val_len))
                # dataset_train = Subset(dataset['train'], range(train_len))
                val_data = ConcatDataset([val_data, dataset_val])
                train_data = ConcatDataset([train_data, dataset_train])                
                val_loader_set.append(DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator))
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader_set = val_loader_set
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        
class MixedPretrain():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path, input_path):
        data_files = {'train': input_path}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:5%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        input_paths = [
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/acl/acl_anthology.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/legal/legal.txt',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/restaurant/yelp_restaurant.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/review/review.txt',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/phone/phone.txt',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/camera/camera.txt'
        ]
        paths = [
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/acl',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/legal',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/restaurant',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/review',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/phone',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/camera',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/bookcorpus',
            # config.dataset_cache[config.dataset_name],
        ]
        
        for i, path in enumerate(paths):
            paths[i] = os.path.join(path, str(self.block_size))
        
        if not config.preprocessed:
            for path, input_path in zip(paths, input_paths):
                self.preprocess(config, path, input_path)
        
        datasets = []
        for path in paths:
            datasets.append(load_from_disk(path))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        val_data = None
        train_data = None
        for dataset in datasets:
            if not val_data:
                val_data = Subset(dataset['validation'], range(1600))
                train_data = Subset(dataset['train'], range(250000))
            else:
                dataset_val = Subset(dataset['validation'], range(1600))
                dataset_train = Subset(dataset['train'], range(250000))
                val_data = ConcatDataset([val_data, dataset_val])
                train_data = ConcatDataset([train_data, dataset_train])                
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class BERTPretrain():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path, input_path):
        data_files = {'train': input_path}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:5%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        input_paths = [
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/bookcorpus/bookcorpus.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/wikipedia/wikipedia.txt'            
        ]
        paths = [
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/bookcorpus',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/wikipedia'            
        ]
        
        for i, path in enumerate(paths):
            paths[i] = os.path.join(path, str(self.block_size))
        
        if not config.preprocessed:
            for path, input_path in zip(paths, input_paths):
                self.preprocess(config, path, input_path)
        
        datasets = []
        for path in paths:
            datasets.append(load_from_disk(path))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        val_data = None
        train_data = None
        for dataset in datasets:
            if not val_data:
                val_data = Subset(dataset['validation'], range(1600))
                train_data = Subset(dataset['train'], range(1500000))
            else:
                dataset_val = Subset(dataset['validation'], range(1600))
                dataset_train = Subset(dataset['train'], range(1500000))
                val_data = ConcatDataset([val_data, dataset_val])
                train_data = ConcatDataset([train_data, dataset_train])                
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class BERTPretrainCount():
    def group_texts(self, examples):
        print(examples.keys())
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path, input_path):
        data_files = {'train': input_path}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:5%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        input_paths = [
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/bookcorpus/bookcorpus.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/wikipedia/wikipedia.txt'            
        ]
        paths = [
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/bookcorpus_gram',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/wikipedia_gram'            
        ]
        
        for i, path in enumerate(paths):
            paths[i] = os.path.join(path, str(self.block_size))
        
        if not config.preprocessed:
            for path, input_path in zip(paths, input_paths):
                self.preprocess(config, path, input_path)
        
        datasets = []
        for path in paths:
            datasets.append(load_from_disk(path))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        val_data = None
        train_data = None
        for dataset in datasets:
            if not val_data:
                val_data = Subset(dataset['validation'], range(1600))
                train_data = Subset(dataset['train'], range(500000))
            else:
                dataset_val = Subset(dataset['validation'], range(1600))
                dataset_train = Subset(dataset['train'], range(500000))
                val_data = ConcatDataset([val_data, dataset_val])
                train_data = ConcatDataset([train_data, dataset_train])                
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for d in train_data:
            print(d)
            break
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)



class MixDomain():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path, input_path):
        data_files = {'train': input_path}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:5%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        
        input_paths = [
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/acl/acl_anthology.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/legal/legal.txt',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/restaurant/yelp_restaurant.txt',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/review/review.txt',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/phone/phone.txt',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/camera/camera.txt'
        ]
        paths = [
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/acl',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/legal',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/restaurant',
            '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/pubmed',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/phone',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/camera',
            # '/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/bookcorpus',
            # config.dataset_cache[config.dataset_name],
        ]
        
        for i, path in enumerate(paths):
            paths[i] = os.path.join(path, str(self.block_size))
        
        if not config.preprocessed:
            for path, input_path in zip(paths, input_paths):
                self.preprocess(config, path, input_path)
        
        datasets = []
        for path in paths:
            datasets.append(load_from_disk(path))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        val_data = None
        train_data = None
        for dataset in datasets:
            if not val_data:
                val_data = Subset(dataset['validation'], range(1600))
                train_data = Subset(dataset['train'], range(250000))
            else:
                dataset_val = Subset(dataset['validation'], range(1600))
                dataset_train = Subset(dataset['train'], range(250000))
                val_data = ConcatDataset([val_data, dataset_val])
                train_data = ConcatDataset([train_data, dataset_train])                
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class LegalForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/mychen/ER_TextSpeech/BERT/data/datasets/legal/legal.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:1%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/legal", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lm_datasets_train = torch.utils.data.Subset(lm_datasets['train'], range(208000))
        lm_datasets_val = torch.utils.data.Subset(lm_datasets['validation'], range(1600))
        seed = 45
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        # self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
       
       
       

class BookCorpus():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/mychen/ER_TextSpeech/BERT/data/datasets/bookcorpus/bookcorpus.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:1%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[1%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/bookcorpus", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lm_datasets_train = torch.utils.data.Subset(lm_datasets['train'], range(500000))
        lm_datasets_val = torch.utils.data.Subset(lm_datasets['validation'], range(1600))
        seed = 45
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        # self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        self.train_loader = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
       
       
class Wikipedia():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/mychen/ER_TextSpeech/BERT/data/datasets/wikipedia/wikipedia.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:1%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[1%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/wikipedia", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lm_datasets_train = torch.utils.data.Subset(lm_datasets['train'], range(500000))
        lm_datasets_val = torch.utils.data.Subset(lm_datasets['validation'], range(1600))
        seed = 45
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        # self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        self.train_loader = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator) 
       
        

class ReviewForLM():
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
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/mychen/ER_TextSpeech/BERT/data/datasets/review/review.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split="train[:1%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split="train[5%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config, train_len, val_len):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/review", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        train_data = Subset(lm_datasets['train'], range(train_len))
        val_data = Subset(lm_datasets['validation'], range(val_len))
        
        seed = 45
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class CustomDataset(Dataset):
    def __init__(self, data):
        self._data = data
    
    def __len__(self):
        return len(self._data['input_ids'])
    
    def __getitem__(self, idx):
        sample = {'input_ids': self._data['input_ids'][idx], 'labels': self._data['labels'][idx], 'attention_mask': self._data['attention_mask'][idx]}
        return sample

class ReplayDataset():
    def __init__(self, batch_size, path):
        tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')
        replay_data = torch.load(path, map_location='cpu')        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        
        self.replay_loader = {}
        
        for key, input_ids in replay_data.items():              
            special_tokens_mask = [tokenizer.get_special_tokens_mask(input_id, already_has_special_tokens=True) for input_id in input_ids]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            
            inputs, labels = data_collator.torch_mask_tokens(input_ids, special_tokens_mask)
            attention_mask = torch.ones(inputs.shape[0], inputs.shape[1])
            data = {'input_ids': inputs, 'labels': labels, 'attention_mask': attention_mask}
            
            replay_dataset = CustomDataset(data)
            replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)            
            self.replay_loader[key] = replay_loader                   


if __name__ == "__main__":
    config = BertConfig.from_json_file('/home/archen/cl/config/bert_small_sp.json')
    
    dataset = CameraForLM(config)
