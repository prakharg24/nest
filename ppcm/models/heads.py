import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
from transformers import BertTokenizer,BertModel,AutoConfig

EPSILON = 1e-10

class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size,
            pretrained_model="medium",
            cached_mode=False,
            load_weight=None,
            model_pretrained=None,
            device='cuda'
    ):
        super(Discriminator, self).__init__()

        model_path = f'models/dialoGPT/{pretrained_model}/'
        config = GPT2Config.from_json_file(os.path.join(model_path, 'config.json'))
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if model_pretrained != None:
            self.encoder = model_pretrained
        else:
            self.encoder = load_model(GPT2LMHeadModel(config), model_path+f"{pretrained_model}_ft.pkl", None, verbose=True)
        self.embed_size = config.n_embd

        self.classifier_head = ClassificationHead(
            class_size=class_size,
            embed_size=self.embed_size
        )
        self.cached_mode = cached_mode
        if(load_weight != None):
            self.classifier_head.load_state_dict(torch.load(load_weight))
        self.device = device
        self.class_size = class_size

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        hidden, _ = self.encoder.transformer(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        return logits


class Scorer(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout
    ):
        super(Scorer, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        embedding_dim = self.bert.config.to_dict()['hidden_size']


        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        #text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        #embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        #hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        #hidden = [batch size, hid dim]

        output = self.out(hidden)

        #output = [batch size, out dim]

        return output

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits


def load_model(model, checkpoint, args, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            print('No checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            print('Loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            print('Loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict
