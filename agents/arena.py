import random
import numpy as np
import copy
from dataloader import get_dataset
from agents import AgentNoPlanningBayesian, AgentDummy, AgentMCTS, AgentQLearning, AgentDeepQLearningMLP
from agent_utils import agent_negotiation, get_chunks

random.seed(32)

### Initialize Meta Data Regarding the Negotiations
elements_to_divide = ["Firewood", "Water", "Food"]
priorities = ["Low", "Medium", "High"]
score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}
length_penalty = 0.5
length_limit = 20
fp_scaling_factor = 0.1

num_rounds = 2

all_data = get_dataset('../casino_with_emotions_and_intents_and_proposals.json')

### Initialize and Collect all agents to take part in the negotiations
agent_list = []
agent_id_counter = 0

# for i in range(5):
#     agent_list.append(AgentDummy(score_weightage, length_penalty, agent_id_counter))
#     agent_id_counter += 1

for i in range(10):
    agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].load_model()
    agent_list[-1].set_mode('eval')
    agent_id_counter += 1

for i in range(10):
    agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].load_model()
    agent_list[-1].set_mode('eval')
    agent_id_counter += 1

for i in range(10):
    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].load_model()
    agent_list[-1].set_mode('train')
    agent_id_counter += 1

for i in range(10):
    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].load_model()
    agent_list[-1].set_mode('train')
    agent_id_counter += 1

for i in range(10):
    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].load_model()
    agent_list[-1].set_mode('train')
    agent_id_counter += 1

agent_scores = {ele.id: 0 for ele in agent_list}

for round in range(num_rounds):
    ## do random pairings
    random.shuffle(agent_list)
    chunk_list = list(get_chunks(agent_list, 2))
    print("Pairings for Round %d" % round)
    print([(ele[0].id, ele[1].id) for ele in chunk_list])
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
print("Final Scores")
print(agent_scores)
