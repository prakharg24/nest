import sys
import argparse
import random
import copy
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")
# random.seed(42)
# np.random.seed(42)

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_prefix(conversation):
    ## Hard coded to Extract first 4 dialogues of the conversation
    ## Can be changed to adapt to the problem statement at hand
    return conversation[:4], conversation[4:]

def run_negotiation(agent_tuple, conversation, participant_info, length_limit, act_ag=0):

    record_conversation = []

    ## Reset agent state at the start of conversation
    agent_tuple[act_ag].start_conversation()
    agent_tuple[(act_ag+1)%2].start_conversation()

    ## Setup Agent's Internal States
    speaker_tuple = []
    for dialogue in conversation:
        if dialogue['speaker_id'] not in speaker_tuple:
            agent_id = dialogue['speaker_id']
            agent_tuple[act_ag].set_participant_info(participant_info[agent_id])
            agent_tuple[act_ag].set_name(agent_id)
            act_ag = (act_ag+1)%2
            speaker_tuple.append(agent_id)

    assert len(speaker_tuple) == 2

    ## Certain agents might require the complete conversation from the dataset
    agent_tuple[act_ag].set_conversation(conversation)
    agent_tuple[(act_ag+1)%2].set_conversation(conversation)

    ## Extract the conversation prefix (currently hard-coded) to help start the conversation
    conv_prefix, conv_suffix = extract_prefix(conversation)

    ## Let both the agents go through the conversation prefix one at a time
    agent_tuple[act_ag].step(None, outdialogue=conv_prefix[0], switch='passive')
    record_conversation.append(conv_prefix[0])
    act_ag = (act_ag+1)%2

    for dia, dia_next in zip(conv_prefix, conv_prefix[1:]):
        agent_tuple[act_ag].step(dia, outdialogue=dia_next, switch='passive')
        record_conversation.append(dia_next)
        act_ag = (act_ag+1)%2

    prev_dialog = conv_prefix[-1]
    while True:
        out_dialog = agent_tuple[act_ag].step(prev_dialog, switch='active')
        # print(out_dialog)
        record_conversation.append(out_dialog)
        if(is_terminated(out_dialog) or len(record_conversation) > length_limit):
            break
        act_ag = (act_ag+1)%2
        prev_dialog = out_dialog

    reward_dict = get_reward_dict(agent_tuple, record_conversation)
    # print(reward_dict)

    agent_tuple[0].step_reward(sum(reward_dict[0].values()))
    agent_tuple[1].step_reward(sum(reward_dict[1].values()))

    return reward_dict

def display_topk_agents(agent_order, k, agent_list, agent_scores, header=""):
    table = []
    for agent in agent_order:
        k -= 1
        row_arr = [agent_list[agent].type, agent]

        scores = list(agent_scores[agent].values())
        scores.append(sum(scores))
        # scores = list(map(prettyfloat, scores))
        row_arr.extend(scores)

        table.append(row_arr)
        if k == 0:
            break

    print(tabulate(np.array(table), headers=header, tablefmt="pretty"))

def display_agent_types(agent_order, agent_collection, header=""):
    table = []
    for agent in agent_order:
        row_arr = [agent, None]

        scores = list(agent_collection[agent].values())
        scores.append(sum(scores))
        # scores = list(map(prettyfloat, scores))
        row_arr.extend(scores)

        table.append(row_arr)

    print(tabulate(np.array(table), headers=header, tablefmt="pretty"))

def display_agent_scores(agent_list, agent_scores, num_highest=5, num_lowest=5, collect_similar_agents=True, header_text=""):
    ## Best Agents
    agent_order = [k for k, v in sorted(agent_scores.items(), key=lambda item: sum(item[1].values()))]
    ind_agent_header = ["Agent Type", "Agent ID"]
    ind_agent_header.extend(list(list(agent_scores.values())[0].keys()))
    ind_agent_header.append("Final Score")

    print("\n\n" + header_text)

    print("\n\n")
    print("WORST %d AGENTS" % num_lowest)
    display_topk_agents(agent_order, num_lowest, agent_list, agent_scores, header=ind_agent_header)

    print("\n\n")
    print("BEST %d AGENTS" % num_highest)
    display_topk_agents(agent_order[::-1], num_highest, agent_list, agent_scores, header=ind_agent_header)

    if collect_similar_agents:
        agent_collection = {}
        agent_count = {}
        for agent in agent_scores:
            agent_type = agent_list[agent].type
            if agent_type in agent_collection:
                for k in agent_scores[agent]:
                    agent_collection[agent_type][k] += agent_scores[agent][k]
                    agent_count[agent_type] += 1
            else:
                agent_collection[agent_type] = agent_scores[agent]
                agent_count[agent_type] = 1

        for agent in agent_collection:
            for k in agent_collection[agent]:
                agent_collection[agent][k] = agent_collection[agent][k]/agent_count[agent]

        print("\n\n")
        print("COMBINED SCORE FOR EVERY AGENT TYPE")
        agent_type_order = [k for k, v in sorted(agent_collection.items(), key=lambda item: sum(item[1].values()))]
        display_agent_types(agent_type_order[::-1], agent_collection, header=ind_agent_header)

    print("\n\n")

def run_stadium(agent_list, num_rounds, length_limit):

    agent_scores = {ele.id: get_zero_reward_dict() for ele in agent_list}
    agent_list_indices = list(range(len(agent_list)))
    for round in tqdm(range(num_rounds)):
        ## Random pairings
        random.shuffle(agent_list_indices)
        chunk_list = list(get_chunks(agent_list_indices, 2))
        # print("Round %d" % round)
        # print("Pairings for Round %d" % round)
        # print([(ele[0].id, ele[1].id) for ele in chunk_list])

        for agent_tuple_index in chunk_list:
            if len(agent_tuple_index)!=2:
                continue

            agent_tuple = [agent_list[agent_tuple_index[0]], agent_list[agent_tuple_index[1]]]

            ### choose a random conversation
            conversation, participant_info = get_random_conversation()

            reward_tuple = run_negotiation(agent_tuple, copy.deepcopy(conversation), participant_info, length_limit=length_limit)
            # print(reward_tuple)

            for k in reward_tuple[0]:
                agent_scores[agent_tuple[0].id][k] += reward_tuple[0][k]
            for k in reward_tuple[1]:
                agent_scores[agent_tuple[1].id][k] += reward_tuple[1][k]

    return agent_scores


parser = argparse.ArgumentParser()
parser.add_argument("--case", default="casino", help="Name of Dataset Used")
parser.add_argument("--stadium", default="default", help="Stadium Configuration Identifier")
parser.add_argument("--leng_limit", type=int, default=20, help="Conversation Length Limit")
parser.add_argument("--train_rounds", type=int, default=20, help="Number of Rounds During Training")
parser.add_argument("--test_rounds", type=int, default=20, help="Number of Rounds During Testing")
parser.add_argument("--vb_highest", type=int, default=5, help="Number of Best Agents Shown in Scoreboard")
parser.add_argument("--vb_lowest", type=int, default=5, help="Number of Worst Agents Shown in Scoreboard")
parser.add_argument("--vb_collect", action='store_true', help="Collect Scores of Each Agent Type")
parser.add_argument("--save", action='store_true', help="Save Trained Agents")
parser.add_argument("--load", action='store_true', help="Load Trained Agents")
parser.add_argument("--save_selected", type=int, help="If Only One Agent Should be Saved")


args = parser.parse_args()

sys.path.append(args.case)
from nest_helper import load_agents, get_random_conversation, is_terminated, get_reward_dict, get_zero_reward_dict

agent_list = load_agents(args.stadium)

print("Training Rounds")
train_scores = run_stadium(agent_list, args.train_rounds, args.leng_limit)

# display_agent_scores(agent_list, train_scores, num_highest=args.vb_highest, num_lowest=args.vb_lowest,
#                                                collect_similar_agents=args.vb_collect, header_text="Training Scores")

### Change all agents to eval
for ele in agent_list:
    ele.set_mode('eval')

if args.save:
    if args.save_selected is not None:
        agent_list[args.save_selected].save_model()
    else:
        for agent in agent_list:
            agent.save_model()

if args.load:
    for agent in agent_list:
        agent.load_model()

print("Testing Rounds")
test_scores = run_stadium(agent_list, args.test_rounds, args.leng_limit)

display_agent_scores(agent_list, test_scores, num_highest=args.vb_highest, num_lowest=args.vb_lowest,
                                               collect_similar_agents=args.vb_collect, header_text="Testing Scores")
