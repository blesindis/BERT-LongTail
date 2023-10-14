import torch.nn as nn
from transformer.Transformer import BertModel, BertDecoderModel, BertLayerSaveModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


# origin bert
class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores
    
    
# a failed try to add a decoder to each bert layer to perform layer replay
class BertWithDecoders(nn.Module):
    def __init__(self, config):
        super(BertWithDecoders, self).__init__()
        self.config = config
        self.bert = BertDecoderModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask)
        
        scores = self.head(outputs[0])
        outputs[0] = scores
        
        # replicated_labels = [labels for _ in range(len(outputs))]
        # losses = [self.criterion(output.view(-1, self.config.vocab_size), target.view(-1)) for output, target in zip(outputs, replicated_labels)]
        return outputs
    
    
# a bert that save each layer output and decoder output
class BertWithSavers(nn.Module):
    def __init__(self, config):
        super(BertWithSavers, self).__init__()
        self.config = config
        self.bert = BertLayerSaveModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        output, layer_outputs = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, layer_outputs