import random
from agents import RandomAgentConsiderate, RandomAgentStubborn
from task_utils import NegotiationStarter

### Initialize Meta Data Regarding the Negotiations
elements_to_divide = ["Firewood", "Water", "Food"]
priorities = ["low", "medium", "high"]
score_weightage = {"high" : 5, "medium" : 4, "low" : 3}
length_penalty = 0.5

num_rounds = 10
starter = NegotiationStarter(datafile='prakhar_labels.json')

exit()

### Initialize and Collect all agents to take part in the negotiations
agent1 = RandomAgentConsiderate(score_weightage, length_penalty, 0)
agent2 = MirrorAgentStubborn(score_weightage, length_penalty, 1)


agent_list = [agent1, agent2]

for round in num_rounds:
    ## do random pairings
    agent_list = random.shuffle(agent_list)
    for a1, a2 in zip(agent_list, agent_list[1:]):
        ### Create random priorities for both agents
        random.shuffle(elements_to_divide)
        a1.set_priority({e:p for e, p in zip(elements_to_divide, priorities)})
        random.shuffle(elements_to_divide)
        a2.set_priority({e:p for e, p in zip(elements_to_divide, priorities)})

        ### get some negotiation starting point
        neg_prefix, agent_dict = starter.get_random_negotiation_prefix()

        agent_tuple = [a1, a2]
        agent_dict = {k: v.id for k, v in zip(agent_dict, agent_tuple)}

        act_ag = 0
        for ele in neg_prefix:
            a1.step_passive(ele['emotion'], ele['intent'], ele['proposal'], agent_tuple[act_ag].id)
            a2.step_passive(ele['emotion'], ele['intent'], ele['proposal'], agent_tuple[act_ag].id)
            act_ag = (act_ag+1)%2

        end_deal = False
        while not end_deal:
            agent_tuple[act_ag].step()
            act_ag = (act_ag+1)%2
