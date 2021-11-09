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

import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# ToDos:
# uncomment corresponding model file import
# change loss function
# change evaluation
# change model saved file name

from model import Encoder, Decoder, Seq2Seq
# from model_intent import Encoder, Decoder, Seq2Seq
# from model_emotion import Encoder, Decoder, Seq2Seq


from utils import *

# Should set seeds to reproduce results
# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# labels to index values for one hot encoding
label_to_index = {'elicit-pref':0, 'no-need':1, 'uv-part':2, 'other-need':3, 'showing-empathy':4, 'vouch-fair':5, 'small-talk':6, 'self-need':7, 'promote-coordination':8, 'non-strategic':9, "sadness": 10, "joy": 11, "anger":12, "fear":13, "surprise":14, "love":15}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataloader class for source and target dialogues
# since dialogue lengths are different, the data is stored as a list of numpy arrays
# during training, each sample is converted into a tensor
class dataLoaderClass(Dataset):
    def __init__(self,x,y):
        self.x = x #torch.tensor(x,dtype=torch.float32, device=device)
        self.y = y #torch.tensor(y,dtype=torch.float32, device=device)
        self.len = len(x) #x.shape[0]


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


#Train
# src_train, trg_train = get_src_trg_by_fname(fname_train)
# training_data = dataLoaderClass(src_train, trg_train)
# print('src train shape: ', len(src_train))
# train_iterator = DataLoader(training_data, batch_size=1, shuffle=True)
# print('Loaded training data')

#Test
# src_test, trg_test = get_src_trg_by_fname(fname_test)
# test_data = dataLoaderClass(src_test, trg_test)
# test_iterator = DataLoader(test_data, batch_size=1, shuffle=True)
# print('Loaded test data')


#Valid
src_valid, trg_valid = get_src_trg_by_fname(fname_valid)
valid_data = dataLoaderClass(src_valid, trg_valid)
valid_iterator = DataLoader(valid_data, batch_size=1, shuffle=True)
print('Loaded validation data')


#Model training

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)  # craigslist uses -0.1, 0.1; it depends on the hid dim size

def train(model, iterator, optimizer, criterion_intent, criterion_emotion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        # batch size = 1 
        src_arr = batch[0] # actual intent sequence 
        trg_arr = batch[1] # target intent sequence
        # convert dialogue sample to tensor and change dimension to (seq_len, batch, ohv_len)
        src = torch.tensor(src_arr,dtype=torch.float32, device=device).permute(1,0,2)
        trg = torch.tensor(trg_arr,dtype=torch.float32, device=device).permute(1,0,2)
        
        
        optimizer.zero_grad()
        
        # model returns linear layer and logsoftmax activated outputs and processed prediction outputs respectively
        output, output_pred = model(src, trg) 
        
        
        output_dim = output.shape[-1]
        
        output = output.view(-1, output_dim)
        
        trg = trg.reshape(-1, output_dim)

        # get target emotion indices
        target_emotion = trg[:,10:]
        target_emotion_idx = torch.argmax(target_emotion, dim=1)
        
        # calculate intent and emotion specific losses
        loss_intent = criterion_intent(output[:,0:10], trg[:,0:10])
        loss_emotion = criterion_emotion(output[:,10:], target_emotion_idx)
        
        # combine the losses together
        loss = loss_intent + loss_emotion
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion_intent, criterion_emotion):
    print('evaluating...')
    
    model.eval()
    
    epoch_loss = 0
    fin_targets=[]
    fin_outputs=[]
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src_arr = batch[0] 
            trg_arr = batch[1] 
            src = torch.tensor(src_arr,dtype=torch.float32, device=device).permute(1,0,2)
            trg = torch.tensor(trg_arr,dtype=torch.float32, device=device).permute(1,0,2)

            output, output_pred = model(src, trg, 0) #turn off teacher forcing

            output_dim = output.shape[-1]
            
            output = output.view(-1, output_dim)  # outputs to pass to loss
            output_pred = output_pred.view(-1, output_dim)  # processed predicted outputs for accuracy calculation
            
            trg = trg.reshape(-1, output_dim)
            target_intent = trg[:,0:10]
            target_emotion = trg[:,10:]
            target_emotion_idx = torch.argmax(target_emotion, dim=1)

            loss_intent = criterion_intent(output[:,0:10], trg[:,0:10])
            loss_emotion = criterion_emotion(output[:,10:], target_emotion_idx)
            
            loss = loss_intent + loss_emotion
            
            
            epoch_loss += loss.item()

            fin_targets.extend(trg.cpu().detach().numpy().tolist())
            fin_outputs.extend(output_pred.cpu().detach().numpy().tolist())

    fin_targets = (np.array(fin_targets)).astype(np.bool).tolist()
    fin_outputs = (np.array(fin_outputs)).astype(np.bool).tolist()
    print('fin output shape',np.array(fin_outputs).shape)
    print('fin target shape',np.array(fin_outputs).shape)
    intent_output = fin_outputs[:][0:10]
    emo_output = fin_outputs[:][10:]
    emo_op_idx = np.argmax(emo_output, axis=1)
    intent_target = fin_targets[:][0:10]
    emo_target = fin_targets[:][10:]
    emo_trg_idx = np.argmax(emo_target, axis=1)
    accuracy_emo = accuracy_score(emo_trg_idx, emo_op_idx)
    print(f"Accuracy Score for emotion = {accuracy_emo}")
    accuracy_intent = accuracy_score(intent_target, intent_output)
    print(f"Accuracy Score for intent = {accuracy_intent}")
    f1_score_intent = f1_score(intent_target, intent_output, average = 'samples')
    print(f"F1 Score (Intent) = {f1_score_intent}")
    f1_score_emo = f1_score(emo_trg_idx, emo_op_idx, average = 'micro')
    print(f"F1 Score (Emotion) = {f1_score_emo}")

    accuracy = accuracy_score(fin_targets,fin_outputs)
    f1_score_micro = f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = f1_score(fin_targets, fin_outputs, average='macro')
        
    return epoch_loss / len(iterator), accuracy, f1_score_micro, f1_score_macro

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


INPUT_DIM = 16 # change based on deal marking parameter
OUTPUT_DIM = 16 # change based on deal marking parameter
ENC_EMB_DIM = 16  # assuming one hot vector as embedded inputs
DEC_EMB_DIM = 16  # assuming one hot vector as embedded inputs
HID_DIM = 128
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5



enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())

criterion_intent = nn.BCEWithLogitsLoss()   # Multi Label Classification
criterion_emotion = nn.NLLLoss()   # Multi Class Classification
N_EPOCHS = 10    # should increase the number of epochs
CLIP = 1    # clipping threshold for parameter updates

best_valid_loss = float('inf')

print('Begin Training')
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    # train_loss = train(model, train_iterator, optimizer, criterion_intent, criterion_emotion, CLIP)
    train_loss = train(model, valid_iterator, optimizer, criterion_intent, criterion_emotion, CLIP)
    print('train loss:',train_loss)
    valid_loss, accuracy, f1_score_micro, f1_score_macro = evaluate(model, valid_iterator, criterion_intent, criterion_emotion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/s2s-model1-sep-clfs-sep-loss-attn.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

# model.load_state_dict(torch.load('models/s2s-model1-sep-clfs-sep-loss-attn.pt'))

# test_loss, accuracy, f1_score_micro, f1_score_macro = evaluate(model, test_iterator, criterion_intent, criterion_emotion)

# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
# print(f"Accuracy Score = {accuracy}")
# print(f"F1 Score (Micro) = {f1_score_micro}")
# print(f"F1 Score (Macro) = {f1_score_macro}")
