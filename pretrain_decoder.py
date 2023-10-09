import torch.nn as nn
import base_models
from transformers import BertConfig
from ER_TextSpeech.BERT.Dataset_simp import Wikitext
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim

import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def get_available_cuda_device() -> int:
    max_devs = torch.cuda.device_count()
    for i in range(max_devs):
        try:
            mem = torch.cuda.mem_get_info(i)
        except:
            continue
        if mem[0] / mem[1] > 0.85:
            return i
    return -1


def validate(model, val_loader, accelerator, vocab_size):
    criterion = nn.CrossEntropyLoss()
    gathered_losses = [[] for i in range (5)]

    for i, batch in enumerate(val_loader):
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(**batch)
        replicated_labels = [labels for _ in range(len(outputs))]
        losses = [criterion(output.view(-1, vocab_size), target.view(-1)) for output, target in zip(outputs, replicated_labels)]
        for i, loss in enumerate(losses):
            gathered_losses[i].append(accelerator.gather(loss.repeat(len(batch))))
    
    gathered_losses = [torch.cat(single_losses)[:len(val_loader.dataset)] for single_losses in gathered_losses]
    perplexity = [torch.mean(single_losses) for single_losses in gathered_losses]
    
    return perplexity


def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]


def train(model, num_epochs, dataset, vocab_size):
    train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter("log/" + 'bert')
    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    
    for epoch in range(num_epochs):
        model.train()
        
        """train origin bert (MLM only)"""
        # losses = []
        # for i, batch in enumerate(train_loader):
        #     loss, _ = model(**batch)
            
        #     losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
        #     optimizer.zero_grad()
        #     accelerator.backward(loss)
        #     optimizer.step()
        #     lr_scheduler.step()    
        
        # loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        # loss_valid = validate(model, val_loader, accelerator)
        # loss_test = validate(model, test_loader, accelerator)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')

        # if accelerator.is_local_main_process:
        #     writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        #     writer.add_scalar('perplexity_valid', loss_valid, epoch)
        #     writer.add_scalar('perplexity_test', loss_test, epoch)
        #     writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
        """train bert with decoders"""
        criterion = nn.CrossEntropyLoss()
        gathered_losses = [[] for i in range(5)]
        for i, batch in enumerate(train_loader):
            labels = batch['labels']
            
            outputs = model(**batch)
            optimizer.zero_grad()   
                                                                                 
            replicated_labels = [labels for _ in range(len(outputs))]
            losses = [criterion(output.view(-1, vocab_size), target.view(-1)) for output, target in zip(outputs, replicated_labels)]
            
            for loss in losses:                
                # current are 1 main loss + 8 decoder loss range from layer 2-9 (start from 0)                
                accelerator.backward(loss) # retain_graph here too, because there are multiple losses
            optimizer.step()
            lr_scheduler.step() 
            
        
            for i, loss in enumerate(losses):
                gathered_losses[i].append(accelerator.gather(loss.repeat(config.batch_size)))
            
        print(epoch)
        losses_train = [torch.mean(torch.cat(single_losses)[:len(train_loader.dataset)]) for single_losses in gathered_losses]
        losses_valid = validate(model, val_loader, accelerator, vocab_size)
        losses_test = validate(model, test_loader, accelerator, vocab_size)
        for loss_train, loss_valid, loss_test in zip(losses_train, losses_valid, losses_test): 
            accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')

    accelerator.save_state('./output')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    dataset = Wikitext(config=config)
    
    # model = base_models.BertForMLM(config=config)
    model = base_models.BertWithDecoders(config=config)
    model = nn.DataParallel(model)
    
    train(model=model, num_epochs=20, dataset=dataset, vocab_size=config.vocab_size)