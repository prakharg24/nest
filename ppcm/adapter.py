from transformers import (AdamW,WEIGHTS_NAME, CONFIG_NAME)
from argparse import ArgumentParser
from transformers import GPT2Tokenizer
from itertools import chain
from torch.utils.data import Dataset, DataLoader
import torch
import os
import math
from pprint import pformat
import glob
import jsonlines
from nltk import tokenize
import copy
from tqdm import tqdm

from models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
from utils.helper import load_model, average_distributed_scalar, load_model_recursive
from utils.helper import parse_prefixes
from metric.lm_score import get_ppl

MODEL_INPUTS = ["input_ids", "lm_labels"]
# MODEL_INPUTS = ["input_ids", "lm_labels", "labels"]
EOS_ID = 50256

class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data):
        self.data = data
        self.dataset_len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item
    def __len__(self):
        return self.dataset_len

def collate_fn(data):
    padding = 94
    max_l = max(len(x["input_ids"]) for x in data)
    padded_dataset = {n:[] for n in MODEL_INPUTS}
    padded_dataset["labels"] = []
    for x in data:
        padded_dataset["lm_labels"].append( x["lm_labels"]+ [-100]*(max_l-len(x["lm_labels"]))  )
        padded_dataset["input_ids"].append(x["input_ids"]+ [padding]*(max_l-len(x["input_ids"])))
        padded_dataset["labels"].append(x["labels"])

    for input_name in MODEL_INPUTS:
        padded_dataset[input_name] = torch.tensor(padded_dataset[input_name])
    return padded_dataset

def build_input_from_segments(args, history, reply, tokenizer):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    sequence = [tokenizer.encode(h) + [EOS_ID] for h in history] + [tokenizer.encode(reply)]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + sequence[-1]
    return instance

def convert_label_to_onehot(input_label_arr, max_lengths):
    out_label_arr = []
    for ele, mx in zip(input_label_arr, max_lengths):
        if isinstance(ele, int):
            new_arr = [0 for _ in range(mx)]
            new_arr[ele] = 1
            out_label_arr.extend(new_arr)
        else:
            out_label_arr.extend(ele)

    return out_label_arr

def make_data_loader(args, tokenizer, fname):
    response = []
    with jsonlines.open(fname) as reader:
        for i, obj in enumerate(reader):
            text = " ".join(tokenize.sent_tokenize(obj["hyp"]["PPLM"][0][-1]))
            response.append((obj['conversation']['conversation']+[text], obj['labels']))

    dataset = []
    for (r, l) in tqdm(response):
        label = convert_label_to_onehot(l, [6, 10])
        seq = build_input_from_segments(args, r[:-1], r[-1], tokenizer)
        seq["labels"] = label
        dataset.append(seq)
    train_dataset = DatasetTrain(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    return train_loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_size', type=str, default="medium", help='Size of dialoGPT')
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Max number of tokens")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--kl_weight", type=float, default=0, help="kl constraint for language model")
    parser.add_argument("--load_check_point_adapter",type=str,default="")

    args = parser.parse_args()


    args.model_path = f'models/dialoGPT/{args.model_size}/'
    config = GPT2Config.from_json_file(os.path.join(args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    print("Loading Dataset")
    train_loader = make_data_loader(args, tokenizer, 'results/evaluate/all_discriminators.jsonl')
    print("Dataset Loaded")

    if(args.load_check_point_adapter != ""):
        print("Loading ADAPTERS")
        model = load_model_recursive(GPT2LMHeadModel(config), args.load_check_point_adapter, args, verbose=True)
    else:
        model = load_model_recursive(GPT2LMHeadModel(config), args.model_path+f"{args.model_size}_ft.pkl", args, verbose=True)
    model.to(args.device)

    #optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    parameters_to_update = [p for n, p in model.named_parameters() if "adapter" in str(n)]
    optimizer = AdamW(parameters_to_update, lr=args.lr, correct_bias=True)
    # scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])

    print("Starting Training")
    best_loss = 1e10
    # Training function and trainer
    for epoch in range(args.n_epochs):
        loss_ttl = 0.
        ttl_batches = 0
        for ite, batch in enumerate(tqdm(train_loader)):
            model.train()
            task_mask = batch["labels"]
            batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
            input_ids, lm_labels = batch
            lm_loss = model(input_ids=input_ids, lm_labels=lm_labels, task_mask=task_mask, kl_weight=args.kl_weight)
            loss = lm_loss / args.gradient_accumulation_steps
            loss.backward()
            loss_ttl += loss.item()
            ttl_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if ite % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        batch_loss = loss_ttl/ttl_batches
        print("Epoch : %d, Loss : %f" % (epoch, batch_loss))
        if (batch_loss < best_loss):
            print("New Best Achieved")
            torch.save(model.state_dict(), "models/adapter/final_model.pt")
