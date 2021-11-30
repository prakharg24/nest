import sys
case_study = 'casino'
sys.path.append(case_study)

from nest_helper import get_agents, get_random_conversation, is_terminated, get_reward_dict, get_zero_reward_dict

import random
import numpy as np
import copy
from tabulate import tabulate

random.seed(32)
np.random.seed(32)

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

def display_agent_scores(agent_scores, num_highest=5, num_lowest=5, collect_similar_agents=True, header_text=""):
    ## Best Agents
    headers = list(agent_scores[0].keys())
    agent_scores = {k: v for k, v in sorted(agent_scores.items(), key=lambda item: sum(item[1].values()))}
    print("Best %d Agents" % num_highest)


length_limit = 20

num_rounds_train = 20
num_rounds_test = 10

agent_list = get_agents()
agent_scores = {ele.id: get_zero_reward_dict() for ele in agent_list}

for round in range(num_rounds_train):
    ## Random pairings
    random.shuffle(agent_list)
    chunk_list = list(get_chunks(agent_list, 2))
    print("Training Round %d" % round)
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

display_agent_scores(agent_scores, header_text="Final Scores")
