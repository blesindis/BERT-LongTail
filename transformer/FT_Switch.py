import torch
import torch.nn as nn
from transformer.SwitchLastN import SwitchLastNModel
from transformer.Switch import SwitchBertModel


class BertSwitchForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = SwitchBertModel(config)
        # The classification head will be a linear layer that takes the pooled output of BERT
        # and outputs a vector of size `num_labels`
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
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
        # Get the output from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output for classification
        # pooled_output = outputs.pooler_output
        pooled_output = outputs.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))

        # Returning the loss is useful for training, but during inference, you'll
        # mostly care about the logits
        return loss, logits