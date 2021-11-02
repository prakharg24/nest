import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 256

tokenizer_file_path = './tokenizer_emo'
model_file_path = './model_emo'
## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
#tokenizer = BertTokenizer.from_pretrained(tokenizer_file_path,do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_file_path).to(device)
model.eval()

label = {
  4: "sadness",
  2: "joy",
  0: "anger",
  1: "fear",
  5: "surprise",
  3: "love"
}

def get_emotion_label(utterance):

    print("Actual sentence before tokenization: ",utterance)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    input_ids_t = torch.tensor(tokenizer.encode(utterance, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True)).unsqueeze(0)
    attention_masks_t = torch.tensor([[float(i>0) for i in seq] for seq in input_ids_t])
    sentiment = model(input_ids_t.to(device), attention_mask = attention_masks_t.to(device))
    print('sentiment', sentiment)
    sent = sentiment[0].detach().cpu().numpy()
    pred = np.argmax(sent)
    return label[pred]
