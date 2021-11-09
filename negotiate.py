import random
from agents import RandomAgentConsiderate, RandomAgentStubborn
from task_utils import NegotiationStarter
from task_utils import get_chunks, switch_proposal_perspective, get_proposal_score

random.seed(42)

### Initialize Meta Data Regarding the Negotiations
elements_to_divide = ["Firewood", "Water", "Food"]
priorities = ["Low", "Medium", "High"]
score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}
length_penalty = 0.5

num_rounds = 10
starter = NegotiationStarter(datafile='prakhar_labels.json')

### Initialize and Collect all agents to take part in the negotiations
agent1 = RandomAgentConsiderate(score_weightage, length_penalty, 0)
agent2 = RandomAgentStubborn(score_weightage, length_penalty, 1)

agent_list = [agent1, agent2]

agent_scores = {ele.id: 0 for ele in agent_list}

for round in range(num_rounds):
    ## do random pairings
    random.shuffle(agent_list)
    chunk_list = list(get_chunks(agent_list, 2))
    for agent_tuple in chunk_list:
        if len(agent_tuple)!=2:
            continue
        a1, a2 = agent_tuple

        ### get some negotiation starting point
        neg_prefix, agent_dict = starter.get_random_negotiation_prefix()
        conv_length = len(neg_prefix)

        for ag_name, ag_object in zip(agent_dict, agent_tuple):
            ag_object.set_priority_and_proposal(agent_dict[ag_name])

        agent_indices = {}
        agent_counter = 0
        for ele in agent_dict:
            agent_indices[ele] = agent_counter
            agent_counter += 1

        act_ag = 0
        for ele in neg_prefix:
            act_ag = agent_indices[ele['speaker_id']]
            a1.step_passive(ele['emotion'], ele['intent'], ele['proposal'], agent_tuple[act_ag].id)
            a2.step_passive(ele['emotion'], ele['intent'], ele['proposal'], agent_tuple[act_ag].id)

        act_ag = (act_ag+1)%2
        prev_emotion = neg_prefix[-1]['emotion']
        prev_intent = neg_prefix[-1]['intent']
        prev_proposal= neg_prefix[-1]['proposal']

        end_deal = False
        while not end_deal:
            conv_length += 1
            new_proposal = switch_proposal_perspective(prev_proposal)
            prev_emotion, prev_intent, prev_proposal, end_deal = agent_tuple[act_ag].step(prev_emotion, prev_intent, new_proposal)
            act_ag = (act_ag+1)%2

        conv_length_penalty = length_penalty*conv_length
        agent_scores[agent_tuple[(act_ag+1)%2].id] += get_proposal_score(agent_tuple[(act_ag+1)%2].priorities, prev_proposal, score_weightage) - conv_length_penalty
        new_proposal = switch_proposal_perspective(prev_proposal)
        agent_scores[agent_tuple[act_ag].id] += get_proposal_score(agent_tuple[act_ag].priorities, new_proposal, score_weightage) - conv_length_penalty

print(agent_scores)
