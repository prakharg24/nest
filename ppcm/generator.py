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
from models.wd import weight_decoder
from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Config
from utils.helper import load_classifier, load_model, cut_seq_to_eos, parse_prefixes, load_model_recursive, get_name
from utils.helper import EOS_ID, find_ngrams, dist_score, truncate, pad_sequences, print_loss_matplotlib
from utils.utils_sample import scorer

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def make_header(args,id_starter,knowledge):
    str_title = ""
    str_title += "===================================================\n"
    str_title += f"Model={args.model_size} Iteration={args.num_iterations} Step_size={args.stepsize}\n"
    str_title += "===================================================\n"
    name_experiment = f"Iter={args.num_iterations}_Step={args.stepsize}_Start={id_starter}"
    if(knowledge):
        str_title += f"Knowledge={knowledge}\n"
        str_title += "===================================================\n"
        knol = knowledge.replace(" ","_")
        name_experiment = f"Iter={args.num_iterations}_Know={knol}_Step={args.stepsize}_Start={id_starter}"
    return str_title, name_experiment

def logger_conv_ent(args,conv,enc,id_starter,class2idx,classifier,knowledge=None,gold=None):
    str_title, name_experiment = make_header(args,id_starter,knowledge)
    acc_original = []
    acc_pplm = []
    for turn in conv:
        if(turn['speaker']=="PPLM"):
            str_title += "===================================================\n"
            str_title += "PPLM\n"
            str_title += "===================================================\n"
            hypotesis, acc_pplm, plots_array = scorer(args,turn,classifier,enc,class2idx,knowledge,plot=False,gold=gold)
            str_title += tabulate(hypotesis, headers=['Id', 'Loss','Dist','Label', 'BLEU/F1','Text'], tablefmt='simple',floatfmt=".2f",colalign=("center","center","center","center","left"))
            str_title += "\n"
            print(str_title)
        elif(turn['speaker']=="DGPT"):
            str_title += "===================================================\n"
            str_title += "DGPT\n"
            str_title += "===================================================\n"
            hypotesis_original, acc_original, _ = scorer(args,turn,classifier,enc,class2idx,knowledge,gold=gold)
            str_title += tabulate(hypotesis_original, headers=['Id','Loss','Dist','Label', 'BLEU/F1','Text'], tablefmt='simple',floatfmt=".2f",colalign=("center","center","center","center","left"))
            str_title += "\n"
            loss_original = hypotesis_original[0][1]
            str_title += "===================================================\n"
        else: ## human case
            str_title += f"{turn['speaker']} >>> {turn['text']}\n"
            loss_original = 0

    return acc_pplm, acc_original, hypotesis, hypotesis_original


def evaluate(args,model,enc,classifier,class2idx,device):
    list_starters = parse_prefixes(args,tokenizer=enc,seed=args.seed)
    print("===================================================")
    print(f"Model={args.model_size} Discrim={args.discrim} Iteration={args.num_iterations} Step_size={args.stepsize}")
    print("===================================================")
    global_acc_original, global_acc_PPLM = [], []
    lab = class2idx[args.label_class].replace(" ","_").replace("/","")
    base_path = f"results/evaluate/{args.discrim}_class_{lab}/"
    name = get_name(args,base_path,class2idx)
    mode = 'w'
    if os.path.exists(name):
        num_lines = sum(1 for line in open(name,'r'))
        list_starters = list_starters[num_lines:]
        mode = 'a'
    with jsonlines.open(get_name(args,base_path,class2idx), mode=mode) as writer:
        for id_starter, starter in enumerate(list_starters):
            conversation = []
            for t in starter["conversation"]:
                conversation.append({"speaker":"human", "text":t})

            history = starter["conversation"]
            context_tokens = sum([enc.encode(h) + [EOS_ID] for h in history],[])

            if(args.wd):
                context_tokens = [context_tokens]
                original_sentence, perturb_sentence, _, loss, _ = weight_decoder(model=model, enc=enc,
                                                                                args=args, context=context_tokens,
                                                                                device=device,repetition_penalty=args.repetition_penalty,
                                                                                classifier=classifier.classifier_head,knowledge=starter["knowledge"])
            else:
                context_tokens = [context_tokens for _ in range(args.num_samples)]
                original_sentence, perturb_sentence, _, loss, _ = latent_perturb(model=model, enc=enc,
                                                                                args=args, context=context_tokens,
                                                                                device=device,repetition_penalty=args.repetition_penalty,
                                                                                classifier=classifier.classifier_head,knowledge=starter["knowledge"])
            conversation.append({"speaker":"DGPT","text":original_sentence.tolist()})
            conversation.append({"speaker":"PPLM","text":perturb_sentence.tolist(),"loss":loss})
            acc_pplm, acc_original, hypotesis, hypotesis_original = logger_conv_ent(args,conversation,enc,id_starter,class2idx=class2idx,classifier=classifier,knowledge=starter["knowledge"],gold=starter["gold"])
            global_acc_PPLM.append(acc_pplm)
            global_acc_original.append(acc_original)
            writer.write({"acc":{"DGPT":acc_original,"PPLM":acc_pplm}, "hyp":{"DGPT":hypotesis_original,"PPLM":hypotesis},"conversation":starter})


    print(f"Global Acc original:{np.mean(global_acc_original)} Acc PPLM:{np.mean(global_acc_PPLM)}")
    print()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default="medium", help='Size of dialoGPT')
    parser.add_argument('--model_path', '-M', type=str, default='gpt-2_pt_models/dialoGPT/',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument('--discrim', '-D', type=str, default=None,
                        choices=('sentiment',"daily_dialogue_act",
                                 "AG_NEWS"),
                        help='Discriminator to use for loss-type 2')
    parser.add_argument('--label_class', type=int, default=-1, help='Class label used for the discriminator')
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
    parser.add_argument('--grad_length', type=int, default=10000)
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate from the modified latents')
    parser.add_argument('--horizon_length', type=int, default=1, help='Length of future to optimize over')
    parser.add_argument('--window_length', type=int, default=0,
                        help='Length of past which is being optimizer; 0 corresponds to infinite window length')
    # parser.add_argument('--force-token', action='store_true', help='no cuda')
    parser.add_argument('--decay', action='store_true', help='whether to decay or not')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument("--max_history", type=int, default=-1)
    parser.add_argument('--wd', action='store_true', help='greedy based on rev comments')
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
        print("LOADING ADAPTER CONFIG FILE AND INTERACTIVE SCRIPT")
        from models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
    else:
        from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Config

    device = 'cpu' if args.nocuda else 'cuda'
    args.model_path = f'models/dialoGPT/{args.model_size}/'
    config = GPT2Config.from_json_file(os.path.join(args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    if(args.load_check_point_adapter != "None"):
        print("Loading ADAPTERS")
        model = load_model_recursive(GPT2LMHeadModel(config,default_task_id=args.task_id), args.load_check_point_adapter, args, verbose=True)
    else:
        model = load_model(GPT2LMHeadModel(config), args.model_path+f"{args.model_size}_ft.pkl", args, verbose=True)
    model.to(device).eval()

    # Freeze Models weights
    for param in model.parameters():
        param.requires_grad = False

    classifier, class2idx = load_classifier(args, model)

    ## set args.num_iterations to 0 to run the adapter
    ## set args.num_iterations to 50 to run WD
    evaluate(args,model,tokenizer,classifier,class2idx,device)
