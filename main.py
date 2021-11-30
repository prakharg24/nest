import sys
import argparse
import random
import copy
from tabulate import tabulate

random.seed(32)

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_prefix(conversation):
    return conversation[:4], conversation[4:]

def agent_negotiation(agent_tuple, conversation, participant_info, length_limit, act_ag=0):

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
        record_conversation.append(out_dialog)
        if(is_terminated(out_dialog) or len(record_conversation) > length_limit):
            break
        act_ag = (act_ag+1)%2
        prev_dialog = out_dialog

    reward_dict = get_reward_dict(agent_tuple, record_conversation)

    agent_tuple[0].step_reward(sum(reward_dict[0].values()))
    agent_tuple[1].step_reward(sum(reward_dict[1].values()))

    return reward_dict

def display_agent_scores(agent_list, agent_scores, num_highest=5, num_lowest=5, collect_similar_agents=True, header_text=""):
    ## Best Agents
    headers = list(agent_scores[0].keys())
    agent_order = [k for k, v in sorted(agent_scores.items(), key=lambda item: sum(item[1].values()))]
    print("Best %d Agents" % num_highest)
    table = []
    for agent in agent_order[::-1]:
        num_highest -= 1
        row_arr = [agent_list[agent].type]
        row_arr.extend(list(agent_scores[agent].values()))
        table.append(row_arr)
        if num_highest == 0:
            break
    print(tabulate(table))

def run_negotiation(agent_list, num_rounds, length_limit):

    agent_scores = {ele.id: get_zero_reward_dict() for ele in agent_list}
    for round in range(num_rounds):
        ## Random pairings
        random.shuffle(agent_list)
        chunk_list = list(get_chunks(agent_list, 2))
        print("Round %d" % round)
        # print("Pairings for Round %d" % round)
        # print([(ele[0].id, ele[1].id) for ele in chunk_list])

        for agent_tuple in chunk_list:
            if len(agent_tuple)!=2:
                continue

            ### choose a random conversation
            conversation, participant_info = get_random_conversation()

            reward_tuple = agent_negotiation(agent_tuple, copy.deepcopy(conversation), participant_info, length_limit=length_limit)

            for k in reward_tuple[0]:
                agent_scores[agent_tuple[0].id][k] += reward_tuple[0][k]
            for k in reward_tuple[1]:
                agent_scores[agent_tuple[1].id][k] += reward_tuple[1][k]

    return agent_scores


parser = argparse.ArgumentParser()
parser.add_argument("--case", default="casino", help="Name of Dataset Used")
parser.add_argument("--leng_limit", type=int, default=20, help="Conversation Length Limit")
parser.add_argument("--train_rounds", type=int, default=20, help="Number of Rounds During Training")
parser.add_argument("--test_rounds", type=int, default=20, help="Number of Rounds During Testing")


args = parser.parse_args()

sys.path.append(args.case)
from nest_helper import get_agents, get_random_conversation, is_terminated, get_reward_dict, get_zero_reward_dict

agent_list = get_agents()

print("Training")
train_scores = run_negotiation(agent_list, args.train_rounds, args.leng_limit)

display_agent_scores(agent_list, train_scores, header_text="Training Scores")

### Change all agents to eval
for ele in agent_list:
    ele.set_mode('eval')

print("Testing")
test_scores = run_negotiation(agent_list, args.test_rounds, args.leng_limit)

display_agent_scores(agent_list, test_scores, header_text="Testing Scores")
