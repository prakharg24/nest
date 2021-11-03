import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

class BERTMultiLabel(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTMultiLabel, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 256

curr_dir = os.path.dirname(__file__)

model_file_intent = os.path.join(curr_dir, 'models/intent_classifier.pt')
model_file_emotion = os.path.join(curr_dir, 'models/emotion_classifier.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_intent = BERTMultiLabel(num_labels=10).to(device)
model_intent.load_state_dict(torch.load(model_file_intent))
model_intent.eval()

model_emotion = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(device)
model_emotion.load_state_dict(torch.load(model_file_emotion))
model_emotion.eval()

label_emotion = {4: "sadness", 2: "joy", 0: "anger", 1: "fear", 5: "surprise", 3: "love"}
label_intent = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

def get_emotion_label(utterance):
    input_ids_t = torch.tensor(tokenizer.encode(utterance,add_special_tokens=True,max_length=MAX_LEN,truncation=True,padding='max_length')).unsqueeze(0)
    attention_masks_t = torch.tensor([[float(i>0) for i in seq] for seq in input_ids_t])
    sentiment = model_emotion(input_ids_t.to(device), attention_mask = attention_masks_t.to(device))

    sent = sentiment[0].detach().cpu().numpy()
    pred = np.argmax(sent)
    return label_emotion[pred], pred, sent

def get_intent_label(utterance):
    input_ids_t = torch.tensor(tokenizer.encode(utterance,add_special_tokens=True,max_length=MAX_LEN,truncation=True,padding='max_length')).unsqueeze(0)
    attention_masks_t = torch.tensor([[float(i>0) for i in seq] for seq in input_ids_t])
    intent = model_intent(input_ids_t.to(device), attention_masks_t.to(device), None)

    intent = intent[0].detach().cpu().numpy()
    pred = np.argwhere(intent>=0.5)
    pred = [e[0] for e in pred]
    labels = [label_intent[e] for e in pred]

    return labels, pred, intent
