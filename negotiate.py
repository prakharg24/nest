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
length_limit = 20

num_rounds = 10
starter = NegotiationStarter(datafile='prakhar_labels.json')

### Initialize and Collect all agents to take part in the negotiations
agent_list = []

agent_id_counter = 0
for i in range(2):
    agent_list.append(RandomAgentConsiderate(score_weightage, length_penalty, agent_id_counter))
    agent_id_counter += 1

for i in range(10):
    agent_list.append(RandomAgentStubborn(score_weightage, length_penalty, agent_id_counter))
    agent_id_counter += 1

agent_scores = {ele.id: 0 for ele in agent_list}

for round in range(num_rounds):
    ## do random pairings
    random.shuffle(agent_list)
    chunk_list = list(get_chunks(agent_list, 2))
    # print("Pairings for Round %d" % round)
    # print([(ele[0].id, ele[1].id) for ele in chunk_list])
    for agent_tuple in chunk_list:
        if len(agent_tuple)!=2:
            continue

        ### get some negotiation starting point
        neg_prefix, agent_dict = starter.get_random_negotiation_prefix()
        conv_length = len(neg_prefix)

        for ag_name, ag_object in zip(agent_dict, agent_tuple):
            ag_object.set_priority_and_proposal(agent_dict[ag_name])

        a1, a2 = agent_tuple
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
            if (conv_length > length_limit):
                break
            new_proposal = switch_proposal_perspective(prev_proposal)
            prev_emotion, prev_intent, prev_proposal, end_deal = agent_tuple[act_ag].step(prev_emotion, prev_intent, new_proposal)
            act_ag = (act_ag+1)%2

        if (conv_length > length_limit):
            continue

        conv_length_penalty = length_penalty*conv_length
        reward1 = get_proposal_score(agent_tuple[(act_ag+1)%2].priorities, prev_proposal, score_weightage) - conv_length_penalty
        agent_scores[agent_tuple[(act_ag+1)%2].id] += reward1

        new_proposal = switch_proposal_perspective(prev_proposal)
        reward2 = get_proposal_score(agent_tuple[act_ag].priorities, new_proposal, score_weightage) - conv_length_penalty
        agent_scores[agent_tuple[act_ag].id] += reward2

agent_scores = {k: v for k, v in sorted(agent_scores.items(), key=lambda item: item[1])}
print("Final Scores")
print(agent_scores)
