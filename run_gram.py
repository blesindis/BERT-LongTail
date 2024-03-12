import os
import json
import torch

import base_models
from transformers import BertConfig
from accelerate import Accelerator
from utils.train_utils import (
    set_seed,
    load_layer_data,
)



BATCH_SIZE = 64
CENTER_MODEL_PATH = "outputs/bert(128)-bs64-3epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, 'centers-2-momoe-transformer.pth')
CONFIG_PATH = 'config/bert_a.json'

# Path to your saved rare n-grams file
n_gram_size = 8
rare_ngrams_path = "rare_ngrams_10000_n4_freq30-200.json"
config = BertConfig.from_json_file(CONFIG_PATH)


models = {
    "momo_lora": "bert(128)-bs64-3epoch-lr3-momo_model_router_common_lora128_router5000",
    "bert": "bert(128)-bs64-3epoch-lr1-bert",
    "moe": "bert(128)-bs64-epoch3-lr3-moe",
    # "momo_full": "bert(128)-bs64-3epoch-lr3-momo_model_router_common_lora768",
}


def get_batches():
    # Load the rare n-grams from the file
    with open(rare_ngrams_path, 'r') as f:
        rare_ngrams = json.load(f)

    input_ids = []
    attention_masks = []
    labels = []
    for k, v in rare_ngrams.items():
        if '2, 0' in k or '50264' in k:
            continue
        data = k.strip("()").split(", ")
        data = [int(d) for d in data]
        input_ids.append(data)
        attention_masks.append([1,1,1,1])
        labels.append([-100, data[1], -100, data[3]])
        # attention_masks.append([1 for _ in range(n_gram_size)])
        # label = data.copy()
        # label[0] = -100
        # label[2] = -100
        # label[4] = -100
        # label[6] = -100
        # labels.append(label)
        
        
    batches = []
    index = 0
    while index < len(input_ids):
        end = index + BATCH_SIZE
        batch = {'input_ids': torch.tensor(input_ids[index:end]), 'attention_mask': torch.tensor(attention_masks[index:end]), 'labels': torch.tensor(labels[index:end])}
        batches.append(batch)
        index = end
    return batches


def load_model(name, path, accelerator):
    center_model = None
    if 'momo' in name:
        center_model = base_models.BertForMLM(config)
        checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
        center_model.load_state_dict(checkpoint)
        center_model = accelerator.prepare(center_model)
        
        centers = load_layer_data(CENTER_PATH)
        
    if 'momo' in name:
        model = base_models.BertWithMoMoModelRouterCommonAttnLargeNew(config, centers)
    elif 'bert' in name:
        model = base_models.BertForMLM(config)
    elif 'moe' in name:
        model = base_models.BertSwitch(config)
    else:
        raise NotImplementedError
    
    checkpoint = torch.load(os.path.join('outputs', path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    model = accelerator.prepare(model)
    return model, center_model


def test(model, batches, accelerator, name, center_model=None):
    losses = []
    for batch in batches:
        batch = {key: value.to('cuda') for key, value in batch.items()}
        with torch.no_grad():
            if 'momo' in name:
                layer_outputs = []             
                hidden_states = center_model.bert.embeddings(batch['input_ids'])
                layer_outputs.append(hidden_states)                    
                for j in range(config.num_hidden_layers):
                    hidden_states = center_model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                    layer_outputs.append(hidden_states)
                loss, _ = model(**batch, routing_states=layer_outputs)
            else:
                loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    losses = torch.cat(losses)[:len(batch)]
    loss = torch.mean(losses)
    return loss


if __name__ == "__main__":
    
    set_seed(42)
    
    accelerator = Accelerator()
    batches = get_batches()
    for k, v in models.items():
        model, center_model = load_model(k, v, accelerator)
        loss = test(model, batches, accelerator, k, center_model)
        print(f'Loss of Model {k} is {loss.item()}')
        del model
