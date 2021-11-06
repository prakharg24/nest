import argparse
import csv
import json
import math
import time
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext.legacy import data as torchtext_data
from torchtext.legacy import datasets
from tqdm import tqdm, trange
import os
from sklearn.metrics import accuracy_score, f1_score
import torchtext
from models.heads import Discriminator

torch.manual_seed(0)
np.random.seed(0)

device = "cuda"
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 128

class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch

def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    if("X_p" in item_info):
        x_p_batch, _ = pad_sequences(item_info["X_p"])
        x_h_batch, _ = pad_sequences(item_info["X_h"])
        x_batch = (x_p_batch,x_h_batch)
    else:
        x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch

def make_anno_dict(anno_arr):
    outdict = {}

    for ele in anno_arr:
        outdict[ele[0]] = ele[1]

    return outdict

def load_dataset(idx2class, pretrained_model, train_iterator, test_iterator,
                 cached=False, detokenize=False, textlabel=True):

    class2idx = {c: i for i, c in enumerate(idx2class)}
    discriminator = Discriminator(
        class_size=len(idx2class),
        pretrained_model=pretrained_model,
        cached_mode=cached
    ).to(device)

    x, y = [], []
    for ele in tqdm(train_iterator):
        if detokenize:
            ele["text"] = TreebankWordDetokenizer().detokenize(ele["text"])
        seq = discriminator.tokenizer.encode(ele["text"])
        seq = torch.tensor(seq, device=device, dtype=torch.long)
        x.append(seq)
        if textlabel:
            ele["label"] = class2idx[ele["label"]]
        y.append(ele["label"])

    train_dataset = Dataset(x, y)

    test_x, test_y = [], []
    for ele in tqdm(test_iterator):
        if detokenize:
            ele["text"] = TreebankWordDetokenizer().detokenize(ele["text"])
        seq = discriminator.tokenizer.encode(ele["text"])
        seq = torch.tensor(seq, device=device, dtype=torch.long)
        test_x.append(seq)
        if textlabel:
            ele["label"] = class2idx[ele["label"]]
        test_y.append(ele["label"])

    test_dataset = Dataset(test_x, test_y)

    return discriminator, train_dataset, test_dataset


def train_epoch(args,data_loader, discriminator, optimizer,
                epoch=0, cached=False, multi_label=False):
    samples_so_far = 0
    discriminator.train_custom()

    if multi_label:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, (input_t, target_t) in tqdm(enumerate(data_loader)):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        if multi_label:
            target_t = target_t.float()
        loss = loss_fn(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

def evaluate_performance(args,data_loader, discriminator, cached=False, multi_label=False):
    discriminator.eval()
    test_loss = 0
    correct = 0
    if multi_label:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    predicted_list = []
    target_list = []
    with torch.no_grad():
        for input_t, target_t in tqdm(data_loader):
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            if multi_label:
                target_t = target_t.float()
            test_loss += loss_fn(output_t, target_t).item()

            target_list.extend(target_t.cpu().detach().numpy().tolist())
            if multi_label:
                predicted_list.extend(torch.sigmoid(output_t).cpu().detach().numpy().tolist())
            else:
                predicted_list.extend(output_t.argmax(dim=1).cpu().detach().numpy().tolist())

    if multi_label:
        predicted_list = np.array(predicted_list) >=0.5
    test_loss /= len(data_loader.dataset)
    accuracy = accuracy_score(target_list, predicted_list)
    F1 = f1_score(target_list, predicted_list, average='macro')

    return test_loss, accuracy, F1

def get_cached_data_loader(dataset, batch_size, discriminator, shuffle=False):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader

def train_discriminator(
        dataset, pretrained_model="medium",
        epochs=10, batch_size=64,
        save_model=False, cached=False, no_cuda=False):
    global device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    multi_label = False

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == 'sentiment':

        idx2class = ["positive", "negative", "very positive", "very negative", "neutral"]

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
        )

        def emotion_to_dict(input_ele):
            return {"text": input_ele[0], "label": input_ele[1]}

        train_iterator = map(vars, train_data)
        test_iterator = map(vars, test_data)

        discriminator, train_dataset, test_dataset = load_dataset(idx2class, pretrained_model,
                                                                  train_iterator, test_iterator,
                                                                  cached=cached, detokenize=True, textlabel=True)

    elif dataset == 'emotion':

        idx2class = ["anger", "fear", "joy", "love", "sadness", "surprise"]

        df_train = pd.read_csv('data/emotion/train.txt', delimiter=';', header=None)
        df_test = pd.read_csv('data/emotion/test.txt', delimiter=';', header=None)
        train_np = np.array(df_train)
        test_np = np.array(df_test)

        def emotion_to_dict(input_ele):
            return {"text": input_ele[0], "label": input_ele[1]}

        train_iterator = map(emotion_to_dict, train_np)
        test_iterator = map(emotion_to_dict, test_np)

        discriminator, train_dataset, test_dataset = load_dataset(idx2class, pretrained_model,
                                                                  train_iterator, test_iterator,
                                                                  cached=cached, detokenize=False, textlabel=True)

    elif dataset == 'intent':

        idx2class = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']
        extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
        multi_label = True

        train_data = json.load(open('data/casino/casino_train.json'))
        test_data = json.load(open('data/casino/casino_test.json'))

        def extract_intent_from_dialogue(inp_data):
            out_iterator = []
            for item in inp_data:
                complete_log = item['chat_logs']
                annotations = make_anno_dict(item['annotations'])

                for i, utterance in enumerate(complete_log):
                    if utterance['text'] in extra_utterances:
                        continue
                    elif utterance['text'] in annotations:
                        labels = annotations[utterance['text']].split(",")
                        label_arr = [int(ann in labels) for ann in idx2class]
                        out_iterator.append({"text": utterance['text'], "label": label_arr})
            return out_iterator

        train_iterator = extract_intent_from_dialogue(train_data)
        test_iterator = extract_intent_from_dialogue(test_data)

        discriminator, train_dataset, test_dataset = load_dataset(idx2class, pretrained_model,
                                                                  train_iterator, test_iterator,
                                                                  cached=cached, detokenize=False, textlabel=False)

    end = time.time()
    print(f"Train:{len(train_dataset)}")
    print(f"Test:{len(test_dataset)}")
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,shuffle=True
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    accuracy_per_epoch = []
    F1_per_epoch = []
    accuracy_per_epoch_train = []
    F1_per_epoch_train = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            args=args,
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            cached=cached,
            multi_label=multi_label
        )

        loss_train, accuracy_train, f1_train = evaluate_performance(
            args=args,
            data_loader=train_loader,
            discriminator=discriminator, cached=cached, multi_label=multi_label
        )
        accuracy_per_epoch_train.append(accuracy_train)
        F1_per_epoch_train.append(f1_train)

        loss, accuracy, f1 = evaluate_performance(
            args=args,
            data_loader=test_loader,
            discriminator=discriminator, cached=cached, multi_label=multi_label
        )
        accuracy_per_epoch.append(accuracy)
        F1_per_epoch.append(f1)

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))
        print(f"TRAIN: Acc {accuracy_train} F1 {f1_train}")
        print(f"TEST: Acc {accuracy} F1 {f1}")
        print()
        # print("\nExample prediction")

        if save_model and accuracy >= max(accuracy_per_epoch):
            print("New Best Achieved. Model Saved")
            torch.save(discriminator.get_classifier().state_dict(),
                       "models/discriminators/DIALOGPT_{}_classifier_head_best.pt".format(dataset,
                                                               epoch + 1))
    print()

    epoch_max_accuracy = accuracy_per_epoch.index(max(accuracy_per_epoch))
    print("Maximum accuracy on test set obtained at epoch", epoch_max_accuracy + 1)
    print(f"TRAIN Minimum loss {epoch_max_accuracy + 1} ACC:{accuracy_per_epoch_train[epoch_max_accuracy]} F1:{F1_per_epoch_train[epoch_max_accuracy]}" )
    print(f"TEST Minimum loss {epoch_max_accuracy + 1} ACC:{accuracy_per_epoch[epoch_max_accuracy]} F1:{F1_per_epoch[epoch_max_accuracy]}" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="sentiment",
                        help="dataset to train the discriminator on."
                             "In case of generic, the dataset is expected"
                             "to be a TSBV file with structure: class \\t text")
    parser.add_argument("--pretrained_model", type=str, default="medium",
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--save_model", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true",
                        help="use to turn off cuda")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
