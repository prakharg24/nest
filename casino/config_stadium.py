from agents import AgentNoPlanningBayesian, AgentCasino, AgentMCTS, AgentQLearning, AgentDeepQLearningMLP, AgentNoPlanningImitation

baseline = "imitation"

def config_bayesian(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_id_counter += 1

    agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/bayesian.pkl')
    agent_id_counter += 1

    return agent_list

def config_imitation(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_id_counter += 1

    agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/imitation/')
    agent_id_counter += 1

    return agent_list

def config_mcts(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/mcts.pkl')
    agent_id_counter += 1

    return agent_list

def config_qlearning(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/qlearning.pkl')
    agent_id_counter += 1

    return agent_list

def config_deepqlearning(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/deepqlearning/')
    agent_id_counter += 1

    return agent_list

def config_all_isolation_test(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/mcts.pkl')
    agent_id_counter += 1

    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/qlearning.pkl')
    agent_id_counter += 1

    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/deepqlearning/')
    agent_id_counter += 1

    return agent_list


def config_all_society_train(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/mcts_society.pkl')
    agent_id_counter += 1

    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/qlearning_society.pkl')
    agent_id_counter += 1

    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('train')
    agent_list[-1].set_model_loc('casino/models/deepqlearning_society/')
    agent_id_counter += 1

    return agent_list

def config_all_test(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    # agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].set_mode('eval')
    # agent_list[-1].load_model('casino/models/bayesian.pkl')
    # agent_id_counter += 1
    #
    # agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].set_mode('eval')
    # agent_list[-1].load_model('casino/models/imitation/')
    # agent_id_counter += 1
    #
    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/mcts_society.pkl')
    agent_id_counter += 1

    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/qlearning_society.pkl')
    agent_id_counter += 1
    #
    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/deepqlearning_society/')
    agent_id_counter += 1

    # agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].set_mode('eval')
    # agent_id_counter += 1
    #
    # agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].set_mode('eval')
    # agent_list[-1].load_model('casino/models/bayesian.pkl')
    # agent_id_counter += 1
    #
    # agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].set_mode('eval')
    # agent_list[-1].load_model('casino/models/imitation/')
    # agent_id_counter += 1

    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/mcts.pkl')
    agent_id_counter += 1

    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/qlearning.pkl')
    agent_id_counter += 1
    #
    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/deepqlearning/')
    agent_id_counter += 1

    # agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
    # agent_list[-1].set_mode('eval')
    # agent_id_counter += 1

    return agent_list

def config_deepqlearningsociety_test(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentDeepQLearningMLP(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/deepqlearning_society/')
    agent_id_counter += 1

    return agent_list

def config_qlearningsociety_test(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentQLearning(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/qlearning_society.pkl')
    agent_id_counter += 1

    return agent_list

def config_mctssociety_test(score_weightage, length_penalty):
    agent_list = []
    agent_id_counter = 0

    if baseline=="dataset":
        agent_list.append(AgentCasino(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_id_counter += 1
    elif baseline=="bayesian":
        agent_list.append(AgentNoPlanningBayesian(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/bayesian.pkl')
        agent_id_counter += 1
    elif baseline=="imitation":
        agent_list.append(AgentNoPlanningImitation(score_weightage, length_penalty, agent_id_counter))
        agent_list[-1].set_mode('eval')
        agent_list[-1].load_model('casino/models/imitation/')
        agent_id_counter += 1
    else:
        raise Exception("Baseline not found")

    agent_list.append(AgentMCTS(score_weightage, length_penalty, agent_id_counter))
    agent_list[-1].set_mode('eval')
    agent_list[-1].load_model('casino/models/mcts_society.pkl')
    agent_id_counter += 1

    return agent_list
