import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from transformers import BertTokenizer

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 256

model_file_intent = 'models/intent_classifier.pt'
model_file_emotion = 'models/emotion_classifier.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_intent = torch.load(model_file_intent)
model_intent.eval()

model_emotion = torch.load(model_file_emotion)
model_emotion.eval()

label_emotion = {4: "sadness", 2: "joy", 0: "anger", 1: "fear", 5: "surprise", 3: "love"}
label_intent = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

def get_emotion_label(utterance):
    input_ids_t = torch.tensor(tokenizer.encode(utterance, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True)).unsqueeze(0)
    attention_masks_t = torch.tensor([[float(i>0) for i in seq] for seq in input_ids_t])
    sentiment = model(input_ids_t.to(device), attention_mask = attention_masks_t.to(device))

    sent = sentiment[0].detach().cpu().numpy()
    pred = np.argmax(sent)
    return label_emotion[pred]

def get_intent_label(utterance):
    input_ids_t = torch.tensor(tokenizer.encode(utterance, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True)).unsqueeze(0)
    attention_masks_t = torch.tensor([[float(i>0) for i in seq] for seq in input_ids_t])
    sentiment = model(input_ids_t.to(device), attention_masks_t.to(device), None)

    sent = sentiment[0].detach().cpu().numpy()
    pred = sent >= 0.5
    return label_intent[pred]
