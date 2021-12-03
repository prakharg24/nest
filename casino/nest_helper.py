import random

from dataloader import get_dataset
from agent_utils import get_proposal_score, switch_proposal_perspective
from config_stadium import config_bayesian, config_imitation, config_mcts, config_qlearning, config_deepqlearning
from config_stadium import config_all_dataset_test, config_all_society_train, config_all_society_test, config_all_test

all_data = get_dataset('casino/casino_with_emotions_and_intents_and_proposals.json')
random.shuffle(all_data)
conv_ind = 0
length_penalty = 0.2
score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}

def load_agents(stadium):
    if stadium=="default":
        stadium = "config_all"

    if stadium=="config_bayesian":
        agent_list = config_bayesian(score_weightage, length_penalty)
    elif stadium=="config_imitation":
        agent_list = config_imitation(score_weightage, length_penalty)
    elif stadium=="config_mcts":
        agent_list = config_mcts(score_weightage, length_penalty)
    elif stadium=="config_qlearning":
        agent_list = config_qlearning(score_weightage, length_penalty)
    elif stadium=="config_deepqlearning":
        agent_list = config_deepqlearning(score_weightage, length_penalty)
    elif stadium=="config_all_dataset_test":
        agent_list = config_all_dataset_test(score_weightage, length_penalty)
    elif stadium=="config_all_society_test":
        agent_list = config_all_society_test(score_weightage, length_penalty)
    elif stadium=="config_all_society_train":
        agent_list = config_all_society_train(score_weightage, length_penalty)
    elif stadium=="config_all_test":
        agent_list = config_all_test(score_weightage, length_penalty)
    else:
        raise ValueError('Stadium Identifier Not Recognised')

    return agent_list

def get_random_conversation():
    global conv_ind
    conversation, participant_info = all_data[conv_ind]
    conv_ind = (conv_ind + 1)%len(all_data)
    return conversation, participant_info

def is_terminated(dialogue):
    if (dialogue['text']=='Accept-Deal' or dialogue['text']=='Walk-Away'):
        return True
    else:
        return False

def get_zero_reward_dict():
    return {"Negotiation Reward" : 0, "Length Penalty" : 0, "Fairness Penalty" : 0}

def get_reward_dict(agent_tuple, record_conversation):
    reward_tuple = [get_zero_reward_dict(), get_zero_reward_dict()]

    reward_tuple[0]['Length Penalty'] = -1*length_penalty*len(record_conversation)
    reward_tuple[1]['Length Penalty'] = -1*length_penalty*len(record_conversation)

    last_agent = (len(record_conversation) + 1)%2

    if record_conversation[-1]['text']=='Accept-Deal' and record_conversation[-2]['proposal'] is not None:
        reverted_proposal = switch_proposal_perspective(record_conversation[-2])

        reward_tuple[last_agent]['Negotiation Reward'] = get_proposal_score(agent_tuple[last_agent].priorities, reverted_proposal['proposal'], agent_tuple[last_agent].score_weightage)
        reward_tuple[(last_agent+1)%2]['Negotiation Reward'] = get_proposal_score(agent_tuple[(last_agent+1)%2].priorities, record_conversation[-2]['proposal'], agent_tuple[(last_agent+1)%2].score_weightage)

        fairness_penalty = 0.1 * abs(reward_tuple[last_agent]['Negotiation Reward'] - reward_tuple[(last_agent+1)%2]['Negotiation Reward'])

        reward_tuple[0]['Fairness Penalty'] = -1*fairness_penalty
        reward_tuple[1]['Fairness Penalty'] = -1*fairness_penalty

    return reward_tuple

def save_agents(agent_list):
    for agent in agent_list:
        agent.save_model()
