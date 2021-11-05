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
from sklearn.metrics import f1_score
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

def load_dataset(idx2class, pretrained_model, train_iterator, test_iterator, cached=False, detokenize=False):

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
        y.append(class2idx[ele["label"]])

    train_dataset = Dataset(x, y)

    test_x, test_y = [], []
    for ele in test_iterator:
        if detokenize:
            ele["text"] = TreebankWordDetokenizer().detokenize(ele["text"])
        seq = discriminator.tokenizer.encode(ele["text"])
        seq = torch.tensor(seq, device=device, dtype=torch.long)
        test_x.append(seq)
        test_y.append(class2idx[ele["label"]])

    test_dataset = Dataset(test_x, test_y)

    return discriminator, train_dataset, test_dataset


def train_epoch(args,data_loader, discriminator, optimizer,
                epoch=0, cached=False):
    samples_so_far = 0
    discriminator.train_custom()

    ce_loss = torch.nn.CrossEntropyLoss()
    for batch_idx, (input_t, target_t) in tqdm(enumerate(data_loader)):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = ce_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

def evaluate_performance(args,data_loader, discriminator, cached=False):
    discriminator.eval()
    test_loss = 0
    correct = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    predicted_list = []
    target_list = []
    with torch.no_grad():
        for input_t, target_t in tqdm(data_loader):
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += ce_loss(output_t, target_t).item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()
            predicted_list.append(pred_t.squeeze().tolist())
            target_list.append(target_t.tolist())

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]), average='micro')


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

        train_iterator = map(vars, train_data)
        test_iterator = map(vars, test_data)

        discriminator, train_dataset, test_dataset = load_dataset(idx2class, pretrained_model,
                                                                  train_iterator, test_iterator,
                                                                  cached=cached, detokenize=True)

    elif dataset == 'emotion':

        idx2class = ["anger", "fear", "joy", "love", "sadness", "surprise"]

        df_train = pd.read_csv('data/emotion/train.txt', delimiter=';', header=None)
        df_test = pd.read_csv('data/emotion/val.txt', delimiter=';', header=None)
        train_np = np.array(df_train)
        test_np = np.array(df_test)

        def emotion_to_dict(input_ele):
            return {"text": input_ele[0], "label": input_ele[1]}

        train_iterator = map(emotion_to_dict, train_np)
        test_iterator = map(emotion_to_dict, test_np)

        discriminator, train_dataset, test_dataset = load_dataset(idx2class, pretrained_model,
                                                                  train_iterator, test_iterator,
                                                                  cached=cached, detokenize=False)

    # elif dataset == 'intent':

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

    loss_per_epoch = []
    accuracy_per_epoch = []
    F1_per_epoch = []

    loss_per_epoch_train = []
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
            cached=cached
        )

        loss_train, accuracy_train, f1_train = evaluate_performance(
            args=args,
            data_loader=train_loader,
            discriminator=discriminator, cached=cached
        )
        loss_per_epoch_train.append(loss_train)
        accuracy_per_epoch_train.append(accuracy_train)
        F1_per_epoch_train.append(f1_train)

        loss, accuracy, f1 = evaluate_performance(
            args=args,
            data_loader=test_loader,
            discriminator=discriminator, cached=cached
        )
        loss_per_epoch.append(loss)
        accuracy_per_epoch.append(accuracy)
        F1_per_epoch.append(f1)

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))
        print(f"TRAIN: Acc {accuracy_train} F1 {f1_train}")
        print(f"TEST: Acc {accuracy} F1 {f1}")
        print()
        # print("\nExample prediction")

        if save_model:
            torch.save(discriminator.get_classifier().state_dict(),
                       "models/discriminators/DIALOGPT_{}_classifier_head_epoch_{}.pt".format(dataset,
                                                               epoch + 1))
    print()
    epoch_min_loss = loss_per_epoch.index(min(loss_per_epoch))
    print(f"TRAIN Minimum loss {epoch_min_loss + 1} ACC:{accuracy_per_epoch_train[epoch_min_loss]} F1:{F1_per_epoch_train[epoch_min_loss]}" )
    print(f"TEST Minimum loss {epoch_min_loss + 1} ACC:{accuracy_per_epoch[epoch_min_loss]} F1:{F1_per_epoch[epoch_min_loss]}" )

    epoch_max_accuracy = accuracy_per_epoch.index(max(accuracy_per_epoch))
    print("Maximum accuracy on test set obtained at epoch", epoch_max_accuracy + 1)
    print(f"TRAIN Minimum loss {epoch_max_accuracy + 1} ACC:{accuracy_per_epoch_train[epoch_max_accuracy]} F1:{F1_per_epoch_train[epoch_max_accuracy]}" )
    print(f"TEST Minimum loss {epoch_max_accuracy + 1} ACC:{accuracy_per_epoch[epoch_max_accuracy]} F1:{F1_per_epoch[epoch_max_accuracy]}" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="sentiment",
                        choices=("sentiment", "clickbait", "toxic",
                                 "daily_dialogue_topics","daily_dialogue_act",
                                 "daily_dialogue_emotion","generic","emocap","NLI","MNLI","DNLI",
                                 "TC_AG_NEWS","TC_SogouNews","TC_DBpedia","TC_YahooAnswers","empathetic_dialogue",
                                 "emotion","pun"),
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
