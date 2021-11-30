import sys
case_study = 'casino'
sys.path.append(case_study)

from nest_helper import get_agents, get_dataset, is_terminated

import random
import numpy as np
import copy

random.seed(32)
np.random.seed(32)

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_prefix(conversation):
    return conversation[:4], conversation[4:]

def agent_negotiation(agent_tuple, conversation, participant_info, length_penalty, length_limit, act_ag=0, reward_sf = 1., fairness_sf=0.1):

    record_conversation = []

    ## Reset agent state at the start of conversation
    agent_tuple[act_ag].start_conversation()
    agent_tuple[(act_ag+1)%2].start_conversation()

    ## Setup Agent's Internal States
    speaker_tuple = []
    for dialogue in conversation:
        if dialogue['speaker_id'] not in speaker_tuple:
            agent_id = dialogue['speaker_id']
            agent_tuple[act_ag].set_priority(participant_info[agent_id])
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
    agent_tuple[act_ag].step_passive(None, conv_prefix[0])
    record_conversation.append(conv_prefix[0])
    act_ag = (act_ag+1)%2

    for dia, dia_next in zip(conv_prefix, conv_prefix[1:]):
        agent_tuple[act_ag].step_passive(dia, dia_next)
        record_conversation.append(dia_next)
        act_ag = (act_ag+1)%2

    prev_dialog = conv_prefix[-1]
    while True:
        out_dialog = agent_tuple[act_ag].step_active(prev_dialog)
        record_conversation.append(out_dialog)
        if(is_terminated(out_dialog) or len(record_conversation) > length_limit):
            break
        act_ag = (act_ag+1)%2
        prev_dialog = out_dialog

    reward_tuple = [0, 0]
    conv_length_penalty = length_penalty * len(record_conversation)
    prev_dialog_reverted = switch_proposal_perspective(prev_dialog)

    term_reward = get_termination_reward(agent_tuple, record_conversation)
    fair_penalty = get_fairness_penalty(agent_tuple, record_conversation)
    ## Add more constraints as required

    reward_tuple[0] += term_reward[0]*reward_sf - conv_length_penalty - fair_penalty[1]*fairness_sf
    reward_tuple[1] += term_reward[1]*reward_sf - conv_length_penalty - fair_penalty[1]*fairness_sf

    agent_tuple[0].step_reward(reward_tuple[0])
    agent_tuple[1].step_reward(reward_tuple[1])

    return record_conversation, reward_tuple

### Initialize Meta Data Regarding the Negotiations
elements_to_divide = ["Firewood", "Water", "Food"]
priorities = ["Low", "Medium", "High"]
score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}
length_penalty = 0.5
length_limit = 20
fp_scaling_factor = 0.

num_rounds = 20
num_rounds_test = 10

all_data = get_dataset('../casino_with_emotions_and_intents_and_proposals.json')

### Initialize and Collect all agents to take part in the negotiations
agent_list = []
agent_id_counter = 0

# for i in range(5):
#     agent_list.append(AgentDummy(score_weightage, length_penalty, agent_id_counter))
#     agent_id_counter += 1

num_copies = 2

for i in range(num_copies):
    agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].load_model()
    agent_list[-1].set_mode('eval')
    agent_id_counter += 1

for i in range(num_copies):
    agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].load_model()
    agent_list[-1].set_mode('eval')
    agent_id_counter += 1

for i in range(num_copies):
    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].load_model()
    agent_list[-1].set_mode('train')
    agent_id_counter += 1

for i in range(num_copies):
    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].load_model()
    agent_list[-1].set_mode('train')
    agent_id_counter += 1
#
for i in range(num_copies):
    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].load_model()
    agent_list[-1].set_mode('train')
    agent_id_counter += 1

agent_scores = {ele.id: 0 for ele in agent_list}

for round in range(num_rounds):
    ## do random pairings
    random.shuffle(agent_list)
    chunk_list = list(get_chunks(agent_list, 2))
    print("Training Round %d" % round)
    # print("Pairings for Round %d" % round)
    # print([(ele[0].id, ele[1].id) for ele in chunk_list])
    for agent_tuple in chunk_list:
        if len(agent_tuple)!=2:
            continue

        ### choose a random conversation
        conv_ind = np.random.choice(range(len(all_data)))
        (conversation, participant_info) = all_data[conv_ind]

        reward_tuple = agent_negotiation(agent_tuple, copy.deepcopy(conversation), participant_info, length_penalty, score_weightage, length_limit=20, act_ag=0, fp_scaling_factor=0.1)
        agent_scores[agent_tuple[0].id] += reward_tuple[0]
        agent_scores[agent_tuple[1].id] += reward_tuple[1]

# agent_list[-1].save_model()
agent_scores = {k: v for k, v in sorted(agent_scores.items(), key=lambda item: item[1])}
print("Final Training Scores")
print(agent_scores)


for ele in agent_list:
    ele.set_mode('eval')

for ele in agent_scores:
    agent_scores[ele] = 0

for round in range(num_rounds_test):
    ## do random pairings
    random.shuffle(agent_list)
    chunk_list = list(get_chunks(agent_list, 2))
    print("Testing for Round %d" % round)
    # print("Pairings for Round %d" % round)
    # print([(ele[0].id, ele[1].id) for ele in chunk_list])
    for agent_tuple in chunk_list:
        if len(agent_tuple)!=2:
            continue

        ### choose a random conversation
        conv_ind = np.random.choice(range(len(all_data)))
        (conversation, participant_info) = all_data[conv_ind]

        reward_tuple = agent_negotiation(agent_tuple, copy.deepcopy(conversation), participant_info, length_penalty, score_weightage, act_ag=0, length_limit=20, fp_scaling_factor=0.1)
        agent_scores[agent_tuple[0].id] += reward_tuple[0]
        agent_scores[agent_tuple[1].id] += reward_tuple[1]

# agent_list[-1].save_model()
agent_scores = {k: v for k, v in sorted(agent_scores.items(), key=lambda item: item[1])}
print("Final Testing Scores")
print(agent_scores)

for i in range(5):
    local_agent_score_arr = []
    for j in range(num_copies):
        ind = num_copies*i + j
        local_agent_score_arr.append(agent_scores[ind])
    print(local_agent_score_arr)
    print(np.mean(local_agent_score_arr), np.std(local_agent_score_arr))


## Train/Test the Model with the Negotiation Setup

## Save Trained Models if Any
