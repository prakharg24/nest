import pickle
from agents import AgentNoPlanningBayesian, AgentDataset, AgentMCTS, AgentQLearning, AgentDeepQLearningMLP, AgentNoPlanningImitation

## List of Agent Classes, their count, training/eval mode and checkpoints if any
agent_class = [AgentNoPlanningBayesian, AgentDataset]
agent_count = [1, 1]
agent_mode = ['train', 'eval']
agent_ckpt = [None, None]

assert len(agent_class) == len(agent_count) == len(agent_mode) == len(agent_ckpt)

## Converting individual lists to a single list of dictionaries
agent_list = []
for i in range(len(agent_list)):
    agent_dict = {}
    agent_dict['class'] = agent_class[i]
    agent_dict['count'] = agent_count[i]
    agent_dict['mode'] = agent_mode[i]
    agent_dict['ckpt'] = agent_ckpt[i]
    agent_list.append(agent_dict)

fileout = open('casino_config.pkl', 'wb')

pickle.dump(agent_list, fileout, pickle.HIGHEST_PROTOCOL)
