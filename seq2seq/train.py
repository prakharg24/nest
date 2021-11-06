import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,matthews_corrcoef,f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
import sys
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from bertclassifier.inference import get_emotion_label, get_intent_label
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model import Encoder, Decoder, Seq2Seq

label_to_index = {'elicit-pref':0, 'no-need':1, 'uv-part':2, 'other-need':3, 'showing-empathy':4, 'vouch-fair':5, 'small-talk':6, 'self-need':7, 'promote-coordination':8, 'non-strategic':9, "sadness": 10, "joy": 11, "anger":12, "fear":13, "surprise":14, "love":15}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse(utterance):
        emotion_label, emotion_index, emotion_logits = get_emotion_label(utterance)
        intent_labels, intent_indices, intent_logits = get_intent_label(utterance)

        emotion_dict = {'label' : emotion_label, 'index' : emotion_index, 'logits' : emotion_logits}
        intent_dict = {'label' : intent_labels, 'index' : intent_indices, 'logits' : intent_logits}

        return {'emotion': emotion_dict, 'intent': intent_dict}

def make_anno_dict(anno_arr):
    outdict = {}

    for ele in anno_arr:
        outdict[ele[0]] = ele[1]

    return outdict

def get_dialogs_from_json(fname):
    print('Loading dialogues from file')
    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
    annotation_list = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']
    max_length = 0

    data = json.load(open(fname))

    dialogue_utterances = {}

    for item in data:
        X = []
        num_of_utterances = 0
        complete_log = item['chat_logs']
        annotations = make_anno_dict(item['annotations'])

        for i, utterance in enumerate(complete_log):
            if utterance['text'] in extra_utterances:
                continue
            elif utterance['text'] in annotations:
                X.append((utterance['text']))
                num_of_utterances+=1
        if num_of_utterances > max_length:
          max_length = num_of_utterances  
        dialogue_utterances[ item['dialogue_id'] ] = X

    return dialogue_utterances, max_length


def get_one_hot_encoding(utterance_labels):
  ohv = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for label in utterance_labels:
    if type(label['label']) is str:
      ohv[label_to_index[label['label']]] = 1
    else :
      for lbl in label['label']:
        ohv[label_to_index[lbl]] = 1
  return ohv

def get_src_trg_by_fname(fname):
  dialogue_utterances, max_dialogue_length = get_dialogs_from_json(fname)
  ohv_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  print('Creating one hot representation of labels')
  src = []
  trg = []
  for dialogue_id, utterances in dialogue_utterances.items():
    src_per_dialogue = []
    trg_per_dialogue = []
    ohv_utterances_in_dialogue = []
    for utterance in utterances:
      label = parse(utterance)
      utterance_labels = []
      utterance_labels.append(label['emotion'])
      utterance_labels.append(label['intent'])
      ohv = get_one_hot_encoding(utterance_labels)
      ohv_utterances_in_dialogue.append(ohv)
    if len(ohv_utterances_in_dialogue) < max_dialogue_length:
      diff = max_dialogue_length - len(ohv_utterances_in_dialogue)
      for i in range(0, diff):
        ohv_utterances_in_dialogue.append(ohv_pad)
    src_per_dialogue = ohv_utterances_in_dialogue
    trg_per_dialogue = ohv_utterances_in_dialogue[1:]
    trg_per_dialogue.append( ohv_pad )
    src.append(src_per_dialogue)
    trg.append(trg_per_dialogue)
  src = np.array(src)
  # src = np.transpose(src, (1, 0, 2))
  trg = np.array(trg)
  # trg = np.transpose(trg, (1, 0, 2))
  return src, trg

class dataLoaderClass(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32, device=device)
        self.y = torch.tensor(y,dtype=torch.float32, device=device)
        self.len = x.shape[0]


    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

#######################################
### Code Starts Here
#######################################

fname_train = '../bertclassifier/data/casino/casino_train.json'
fname_test = '../bertclassifier/data/casino/casino_test.json'
fname_valid = '../bertclassifier/data/casino/casino_valid.json'

## create dataloaders for train, val, test

#Train
src_train, trg_train = get_src_trg_by_fname(fname_train)
training_data = dataLoaderClass(src_train, trg_train)
# print('src train shape: ', src_train.shape)
train_iterator = DataLoader(training_data, batch_size=32, shuffle=True)
print('Loaded training data')

#Test
src_test, trg_test = get_src_trg_by_fname(fname_test)
test_data = dataLoaderClass(src_test, trg_test)
# print('src test shape: ', src_test.shape)
test_iterator = DataLoader(test_data, batch_size=10, shuffle=True)
print('Loaded test data')

#Valid
src_valid, trg_valid = get_src_trg_by_fname(fname_valid)
valid_data = dataLoaderClass(src_valid, trg_valid)
# print('src valid shape: ', src_valid.shape)
valid_iterator = DataLoader(valid_data, batch_size=10, shuffle=True)
print('Loaded validation data')

# for i, batch in enumerate(valid_iterator):
        
#         #print('Batch: ', batch)
#         srctest = batch[0].permute(1,0,2).to(device)  # actual intent sequence 
#         trgtest = batch[1].permute(1,0,2).to(device)  # target intent sequence
#         print('src in train', srctest.size())
#         print('trg in train', trgtest.size())

#Model training

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)  # craigslist uses -0.1, 0.1

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        #print('Batch: ', batch)
        # before permute function, src.size(), trg.size() --> (batch_size, dialogue_length, ohv_size)
        # after permute function, src.size(), trg.size() --> (dialogue_length, batch_size, ohv_size)
        src = batch[0].permute(1,0,2).to(device)  # actual intent sequence 
        trg = batch[1].permute(1,0,2).to(device)  # target intent sequence
        # print('src in train', src.size())
        # print('trg in train', trg.size())
        
        optimizer.zero_grad()
        
        output = model(src, trg) 
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, ohv_len]
        
        output_dim = output.shape[-1]
        # print('output_dim:', output_dim)
        
        # output = output[1:].view(-1, output_dim)
        output = output.view(-1, output_dim)
        # print('after output view selection:', output.size())
        # trg = trg[1:].view(-1)
        # print('target size:', trg.size())
        # trg.contiguous()
        trg = trg.reshape(-1, output_dim)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    print('evaluating...')
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0].permute(1,0,2).to(device)
            trg = batch[1].permute(1,0,2).to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            # output = output[1:].view(-1, output_dim)
            # trg = trg[1:].view(-1)
            output = output.view(-1, output_dim)
            trg = trg.reshape(-1, output_dim)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


INPUT_DIM = 16 # change based on deal marking parameter
OUTPUT_DIM = 16 # change based on deal marking parameter
ENC_EMB_DIM = 16  #original is 5
DEC_EMB_DIM = 16  #original is 5
HID_DIM = 128
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5



enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# print(model.summary())
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

print('Begin Training')
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    # train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    train_loss = train(model, valid_iterator, optimizer, criterion, CLIP)
    # print('train loss:',train_loss)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/s2s-model1.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('models/s2s-model1.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
