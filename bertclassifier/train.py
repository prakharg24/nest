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

def make_anno_dict(anno_arr):
    outdict = {}

    for ele in anno_arr:
        outdict[ele[0]] = ele[1]

    return outdict

def get_dialogs_from_json(fname):

    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
    annotation_list = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']

    data = json.load(open(fname))
    X = []
    Y = []

    for item in data:
        complete_log = item['chat_logs']
        annotations = make_anno_dict(item['annotations'])

        for i, utterance in enumerate(complete_log):
            if utterance['text'] in extra_utterances:
                continue
            elif utterance['text'] in annotations:
                labels = annotations[utterance['text']].split(",")
                label_arr = []
                for ann in annotation_list:
                    if ann in labels:
                        label_arr.append(1)
                    else:
                        label_arr.append(0)
                X.append((utterance['text']))
                Y.append(label_arr)

    return X, Y

def get_emotion_sentences(fname):
    df = pd.read_csv(fname, delimiter=';', header=None, names=['sentence','label'])

    labelencoder = LabelEncoder()
    df['label_enc'] = labelencoder.fit_transform(df['label'])

    df[['label','label_enc']].drop_duplicates(keep='first')

    df.rename(columns={'label':'label_desc'},inplace=True)
    df.rename(columns={'label_enc':'label'},inplace=True)

    sentences = df.sentence.values
    labels = df.label.values

    return sentences, labels

class BERTMultiLabel(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTMultiLabel, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def eval_multilabel(model, validation_dataloader):
    fin_targets=[]
    fin_outputs=[]
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, b_input_mask, None)

        fin_targets.extend(b_labels.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = f1_score(fin_targets, fin_outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")


def eval_multiclass(model, validation_dataloader):
    eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits[0].to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        tmp_eval_accuracy = accuracy_score(labels_flat,pred_flat)
        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)
        eval_accuracy += tmp_eval_accuracy
        eval_mcc_accuracy += tmp_eval_mcc_accuracy
        nb_eval_steps += 1

    print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')
    print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')

#######################################
### Code Starts Here
#######################################
dataset = sys.argv[1]

# identify and specify the GPU as the device, later in training loop we will load data into device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset == 'casino':
    sentences, labels = get_dialogs_from_json('data/casino/casino.json')
elif dataset == 'emotion':
    sentences, labels = get_emotion_sentences('data/emotion/emotion.txt')
else:
    raise ValueError('Dataset not defined')

MAX_LEN = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN,truncation=True,padding='max_length') for sent in sentences]

attention_masks = []
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

train_inputs,validation_inputs,train_labels,validation_labels = train_test_split(input_ids,labels,random_state=41,test_size=0.1)
train_masks,validation_masks,_,_ = train_test_split(attention_masks,input_ids,random_state=41,test_size=0.1)

# convert all our data into torch tensors, required data type for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 16

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory
train_data = TensorDataset(train_inputs,train_masks,train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)

validation_data = TensorDataset(validation_inputs,validation_masks,validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler,batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
if dataset == 'casino':
    model = BERTMultiLabel(num_labels=10).to(device)
elif dataset == 'emotion':
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(device)

# Parameters:
lr = 2e-5
adam_epsilon = 1e-8

# Number of training epochs (authors recommend between 2 and 4)
epochs = 1

num_warmup_steps = 0
num_training_steps = len(train_dataloader)*epochs

### In Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon,correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

# Gradients gets accumulated by default
model.zero_grad()

# tnrange is a tqdm wrapper around the normal python range
for _ in range(1, epochs+1):
    print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
    batch_loss = 0

    for step, batch in enumerate(tqdm(train_dataloader)):
        model.train()

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        if dataset=='casino':
            outputs = model(b_input_ids, b_input_mask, None)
            loss = loss_fn(outputs, b_labels.float())
        elif dataset=='emotion':
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        batch_loss += loss.item()

        if(step==10):
            break

    avg_train_loss = batch_loss / len(train_dataloader)
    print(F'\n\tAverage Training loss: {avg_train_loss}')

    model.eval()

    if dataset=='casino':
        eval_multilabel(model, validation_dataloader)
    elif dataset=='emotion':
        eval_multiclass(model, validation_dataloader)

if dataset=='casino':
    model_save_path = 'models/intent_classifier.pt'
elif dataset=='emotion':
    model_save_path = 'models/emotion_classifier.pt'
torch.save(model, model_save_path)
