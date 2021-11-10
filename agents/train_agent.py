from dataloader import get_dataset
from agents import AgentNoPlanningBayesian, AgentDummy

import random
random.seed(62)

def break_conversation(conversation):
    return conversation[:4], conversation[4:]

def agent_negotiation(agent_tuple, conversation, act_ag=0, mode='train', length_limit=20):
    conv_prefix, conv_suffix = break_conversation(conversation)

    conv_length = 0

    agent_tuple[act_ag].step_passive(None, conv_prefix[0], mode=mode)
    act_ag = (act_ag+1)%2
    conv_length += 1

    for dia, dia_next in zip(conv_prefix, conv_prefix[1:]):
        agent_tuple[act_ag].step_passive(dia, dia_next, mode=mode)
        act_ag = (act_ag+1)%2
        conv_length += 1

    prev_dialog = conv_prefix[-1]
    while True:
        if(conv_length > length_limit):
            break
        out_dialog = agent_tuple[act_ag].step(prev_dialog, mode=mode)
        # print(out_dialog)
        if(out_dialog['text']=='Accept-Deal' or out_dialog['text']=='Walk-Away'):
            break
        conv_length += 1
        act_ag = (act_ag+1)%2
        prev_dialog = out_dialog

    ## How do we get here?????
    # return proposal_tuple, conv_length
    return

def train(agent1, agent2):
    ## Load Dataset
    all_data = get_dataset('../casino_with_emotions.json', shuffle=True)

    ## Go through all the dialogues
    for i, (conversation, participant_info) in enumerate(all_data):
        ## Setup Agent's priority
        ## Assumption here that the first person doesn't speak twice in a row at the start
        agent1_id = conversation[0]['speaker_id']
        agent1.set_priority(participant_info[agent1_id]['value2issue'])
        agent1.set_name(conversation[0]['speaker_id'])

        agent2_id = conversation[1]['speaker_id']
        agent2.set_priority(participant_info[agent2_id]['value2issue'])
        agent2.set_name(conversation[1]['speaker_id'])

        ## Certain agents require complete conversation for training
        agent1.set_conversation(conversation)
        agent2.set_conversation(conversation)

        if(i<20):
            agent_negotiation([agent1, agent2], conversation, act_ag=0, mode='train', length_limit=20)
        else:
            agent_negotiation([agent1, agent2], conversation, act_ag=0, mode='eval', length_limit=20)

    ## save file
    agent.save_model(outfile)

if __name__ == "__main__":
    score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}
    length_penalty = 0.5
    agent1 = AgentDummy(score_weightage, length_penalty, 0)
    agent2 = AgentNoPlanningBayesian(score_weightage, length_penalty, 0)
    train(agent1, agent2)
