from comet_ml import Experiment
import jsonlines
import os
import argparse
import torch
import numpy as np
import random
from transformers import GPT2Tokenizer
from tabulate import tabulate
from sklearn.model_selection import ParameterGrid
tabulate.PRESERVE_WHITESPACE = True

from models.heads import Discriminator
from models.pplm import latent_perturb
from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Config
from utils.helper import load_classifier_arr, load_model, cut_seq_to_eos, parse_prefixes, load_model_recursive, get_name
from utils.helper import EOS_ID, find_ngrams, dist_score, truncate, pad_sequences, print_loss_matplotlib
from utils.utils_sample import scorer

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def get_random_label_arr(idx2class_arr, multilabel_arr):
    label_arr = []
    for multilabel, idx2class in zip(multilabel_arr, idx2class_arr):
        if multilabel:
            num_choices = random.choice(range(4))
            label_classes = random.choices(range(len(idx2class)), k=num_choices)
            label_arr.append([1 if e in label_classes else 0 for e in range(len(idx2class))])
        else:
            label_class = random.choice(range(len(idx2class)))
            label_arr.append(label_class)

    return label_arr

def evaluate(args, model, enc, classifier_arr, idx2class_arr, multilabel_arr, device):
    list_starters = parse_prefixes(args,tokenizer=enc,seed=args.seed)

    global_acc_original, global_acc_PPLM = [], []
    name = 'results/evaluate/all_discriminators.jsonl'

    mode = 'w'
    if os.path.exists(name):
        num_lines = sum(1 for line in open(name,'r'))
        list_starters = list_starters[num_lines:]
        mode = 'a'

    with jsonlines.open(name, mode=mode, flush=True) as writer:
        for id_starter, starter in enumerate(list_starters):
            print(id_starter, len(list_starters))

            label_arr = get_random_label_arr(idx2class_arr, multilabel_arr)

            history = starter["conversation"]
            context_tokens = sum([enc.encode(h) + [EOS_ID] for h in history],[])

            context_tokens = [context_tokens for _ in range(args.num_samples)]
            original_sentence, perturb_sentence, loss = latent_perturb(model=model, enc=enc,
                                                                            args=args, context=context_tokens,
                                                                            device=device, repetition_penalty=args.repetition_penalty,
                                                                            classifier_arr=[ele.classifier_head for ele in classifier_arr],
                                                                            multilabel_arr=multilabel_arr,
                                                                            label_arr=label_arr)

            # print(original_sentence)
            # print(perturb_sentence)
            dgpt_out = {"speaker":"DGPT","text":original_sentence.tolist()}
            pplm_out = {"speaker":"PPLM","text":perturb_sentence.tolist(),"loss":loss}
            hypotesis, acc_pplm, plots_array = scorer(args, pplm_out, classifier_arr, enc, idx2class_arr, label_arr, starter["knowledge"], plot=False, gold=starter["gold"])
            hypotesis_original, acc_original, _ = scorer(args, dgpt_out, classifier_arr, enc, idx2class_arr, label_arr, starter["knowledge"], gold=starter["gold"])

            # exit()
            global_acc_PPLM.append(acc_pplm)
            global_acc_original.append(acc_original)
            writer.write({"acc":{"DGPT":acc_original,"PPLM":acc_pplm}, "hyp":{"DGPT":hypotesis_original,"PPLM":hypotesis}, "conversation":starter, "labels":label_arr})


    print(f"Global Acc original:{np.mean(global_acc_original)} Acc PPLM:{np.mean(global_acc_PPLM)}")
    print()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default="medium", help='Size of dialoGPT')
    parser.add_argument('--model_path', '-M', type=str, default='gpt-2_pt_models/dialoGPT/',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument('--stepsize', type=float, default=0.02)
    parser.add_argument('--num_iterations', type=int, default=0)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5555)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.1) #1.1
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gm_scale", type=float, default=0.95)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument('--nocuda', action='store_true', help='no cuda')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate from the modified latents')
    parser.add_argument('--horizon_length', type=int, default=1, help='Length of future to optimize over')
    parser.add_argument('--window_length', type=int, default=0,
                        help='Length of past which is being optimizer; 0 corresponds to infinite window length')
    # parser.add_argument('--force-token', action='store_true', help='no cuda')
    parser.add_argument('--decay', action='store_true', help='whether to decay or not')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument("--max_history", type=int, default=-1)
    parser.add_argument('--bow_scale_weight', type=float, default=20)
    parser.add_argument('--sample_starter', type=int, default=0)
    parser.add_argument('--all_starter', action='store_true', help='selfchat')
    parser.add_argument("--speaker", type=str, default="PPLM")
    parser.add_argument("--load_check_point_adapter", type=str, default="None")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--trial_id", type=int, default=1)
    parser.add_argument("--BCE", type=bool, default=False)
    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if(args.load_check_point_adapter != "None"):
        from models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
    else:
        from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Config

    device = 'cpu' if args.nocuda else 'cuda'
    args.model_path = f'models/dialoGPT/{args.model_size}/'
    config = GPT2Config.from_json_file(os.path.join(args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    if(args.load_check_point_adapter != "None"):
        model = load_model_recursive(GPT2LMHeadModel(config,default_task_id=args.task_id), args.load_check_point_adapter, args, verbose=True)
    else:
        model = load_model(GPT2LMHeadModel(config), args.model_path+f"{args.model_size}_ft.pkl", args, verbose=True)
    model.to(device).eval()

    # Freeze Models weights
    for param in model.parameters():
        param.requires_grad = False

    classifier_arr, idx2class_arr, multilabel_arr = load_classifier_arr(args, model)

    ## set args.num_iterations to 0 to run the adapter
    ## set args.num_iterations to 50 to run WD
    evaluate(args, model, tokenizer, classifier_arr, idx2class_arr, multilabel_arr, device)
