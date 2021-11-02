import os, sys
import random
import torch
import json
import pandas as pd
from matplotlib.pylab import plt
from sklearn.model_selection import KFold

from arguments import ArgumentHandler
from data import DataHandler
from model import ModelHandler
import emotion_extraction

print("Imports Done")

"""
For every hyp combination:
    For every cv split
        Basically: lets do this, restrict this to just one round of CV, somehow.
"""

def select_hyps(hyp_search):
    exceptions = set(['labels'])
    this_hyp = {}

    for hyp, val in hyp_search.items():

        if isinstance(val, list) and (hyp not in exceptions):
            this_hyp[hyp] = random.choice(val)
        else:
            this_hyp[hyp] = val

    return this_hyp

def get_dialogs_from_file(fname):

    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']

    df = pd.read_csv(fname)
    dat = pd.DataFrame.to_dict(df, orient="records")

    all_data = []

    cur_dialog = []
    cur_id = -1

    for i, item in enumerate(dat):

        if((item["DialogueId"] != item["DialogueId"]) or (item["Utterance"] in extra_utterances)):
            continue

        #valid item
        this_id = item["DialogueId"]
        if((this_id != cur_id) and cur_dialog):
            #found new id
            new_dialog = cur_dialog[:]
            all_data.append(new_dialog)
            cur_dialog = []

        if(isinstance(item["Labels"], str)):
            assert isinstance(item["Labels"], str) and len(item["Labels"])>0, i
            this_item = [item["Utterance"], item["Labels"]]
            cur_dialog.append(this_item)
            cur_id = this_id

    if(cur_dialog):
        new_dialog = cur_dialog[:]
        all_data.append(new_dialog)
        cur_dialog = []

    return all_data


def make_anno_dict(anno_arr):
    outdict = {}

    for ele in anno_arr:
        outdict[ele[0]] = ele[1]

    return outdict

def get_dialogs_from_json(fname):

    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']

    data = json.load(open(fname))
    all_data = []

    for item in data:
        complete_log = item['chat_logs']
        annotations = make_anno_dict(item['annotations'])

        curr_dialog = []
        for i, utterance in enumerate(complete_log):
            if utterance['text'] in extra_utterances:
                break
            if utterance['text'] in annotations:
                curr_dialog.append([utterance['text'], annotations[utterance['text']]])

        if(len(curr_dialog)==0):
            continue
        all_data.append(curr_dialog)

    return all_data

def get_all_data(input_files):
    """
    list of dialogues with annotations.
    """
    all_data = []

    for input_file in input_files:
        all_data += get_dialogs_from_json(input_file)

    return all_data

def get_all_data_dummy():
    """
    simple list of dialogues with annotations. To illustrate the expected input format.
    """
    """
    #DUMMY FOR NOW
    all_data = [

        [
            ["Hello!  Let's work together on a deal for these packages, shall we? What are you most interested in?",
            "Promote-coordination,Preference-elicitation"],
            ["Hey! I'd like some more firewood to keep my doggo warm. What do you need?",
            "Required-other,Preference-elicitation"],
        ],
        [
            ["Hello!  Let's work together on a deal for these packages, shall we? What are you most interested in?",
            "Promote-coordination,Preference-elicitation"],
            ["Hey! I'd like some more firewood to keep my doggo warm. What do you need?",
            "Required-other,Preference-elicitation"],
        ],
    ]

    all_data = all_data*100

    return all_data
    """
    pass

def get_cv_generator(all_data, num_folds):
    kf = KFold(n_splits=num_folds)
    train_ratio = 0.95
    X = [[0] for _ in range(len(all_data))]
    for train_index, test_index in kf.split(X):
        train_dev_data = [all_data[ii] for ii in train_index]
        test_data = [all_data[ii] for ii in test_index]
        ll = len(train_dev_data)
        train_data, dev_data = train_dev_data[:int(ll*train_ratio)], train_dev_data[int(ll*train_ratio):]
        yield (train_data, dev_data, test_data)

def write_graph(net, device, dataloader):
    inputs = None
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, _ = batch
        inputs = (x_input_ids, x_input_type_ids, x_input_mask, x_input_feats)
        break

    board_dir = BOARD_DIR
    writer = SummaryWriter(board_dir)
    writer.add_graph(net, inputs)
    writer.close()
    print("GRAPH WRITTEN FOR TENSORBOARD AT: ", board_dir)

def show_summary_func(net, device, dataloader):
    x_input_ids, x_input_type_ids, x_input_mask, x_input_feats = None, None, None, None
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, _ = batch
        break

    print(summary(net, x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, show_input=True))

def train_and_eval(cv_data, hyps, show_summary=False):
    train_data, val_data, test_data = cv_data[0], cv_data[1], cv_data[2]

    #arguments
    args = ArgumentHandler()
    args.update_hyps(hyps)

    #data
    data = DataHandler(args=args)
    data.train_dataloader = data.get_dataloader(train_data, fit_scalar=True, train=True)
    data.dev_dataloader = data.get_dataloader(val_data)
    data.test_dataloader = data.get_dataloader(test_data)

    #modelling
    model = ModelHandler(data=data, args=args)

    if(show_summary):
        show_summary_func(model.model, model.device, data.train_dataloader)

    best_dev_f1, best_ckpt, summary = model.train_model()

    #now evaluate on the best_ckpt
    model.load_model_from_ckpt(best_ckpt)#load_best_ckpt basically.
    #get results
    train_results = model.evaluate_model("train", sample_size=-1)
    dev_results = model.evaluate_model("dev", sample_size=-1)
    test_results = model.evaluate_model("test", sample_size=-1)

    #build output dict for this run:
    output = {}

    output["model"] = best_ckpt
    output["results"] = {
        "train": train_results,
        "dev": dev_results,
        "test": test_results,
    }
    output["training"] = {
        "ckpt_level_summary": summary,
        "best_dev_f1": best_dev_f1
    }

    del args
    del data
    del model

    return output

def get_cv_mean_f1_on_val(cv_results):

    sum_f1 = 0.0
    for item in cv_results:
        sum_f1 += item["dev"]["mean_positive_f1"]

    return sum_f1/len(cv_results)

def merge_cv_results(cv_results):
    """
    Means across CV
    """
    dtypes = ["train", "dev", "test"]
    props_l1 = ["mean_loss", "mean_accuracy", "mean_positive_f1", "UL-A", "Joint-A"]
    props_l2 = ["accuracy", "positive_f1"]

    merged_results = {}

    for dtype in dtypes:
        merged_results[dtype] = {}
        for prop in props_l1:
            summ = 0.0
            for item in cv_results:
                summ += item[dtype][prop]
            merged_results[dtype][prop] = summ/len(cv_results)

        num_labels = len(cv_results[0][dtype]["label_wise"])
        merged_results[dtype]["label_wise"] = [{} for _ in range(num_labels)]
        for i in range(num_labels):
            for prop in props_l2:
                summ = 0.0
                for item in cv_results:
                    summ += item[dtype]["label_wise"][i][prop]
                merged_results[dtype]["label_wise"][i][prop] = summ/len(cv_results)

    return merged_results

def save_everything(all_cv_outputs, ckptfile, cv_wise_file, hyp_wise_file):
    """
    models and results for the cvs, individually and aggregate data.
    """
    #models
    models = all_cv_outputs["models"]
    torch.save(models, ckptfile)

    #cv wise
    cv_wise = {
        "results": all_cv_outputs["results"],
        "training": all_cv_outputs["training"],
    }
    with open(cv_wise_file, 'w') as fp:
        json.dump(cv_wise, fp)

    #hyp level
    hyp_wise = {
        "hyps": all_cv_outputs["hyps"],
        "results": merge_cv_results(all_cv_outputs["results"])
    }
    with open(hyp_wise_file, 'w') as fp:
        json.dump(hyp_wise, fp)

print("Code starts here")
input_files = [
    '../data/casino.json'
]

all_data = get_all_data(input_files)

print("Total number of dialogues in the dataset: ", len(all_data))

total_utts = 0
for item in all_data:
    total_utts += len(item)
print("Total utterances: ", total_utts)


labels = ['showing-empathy', 'promote-coordination', 'elicit-pref', 'non-strategic', 'no-need', 
          'self-need', 'other-need', 'vouch-fair', 'uv-part', 'small-talk']

num_labels = len(labels)

emotion = emotion_extraction.get_emotion_label(all_data[0][1][0])
print('utterence = ', all_data[0][1][0], ' emotion = ', emotion)
