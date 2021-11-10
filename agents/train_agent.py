from dataloader import get_dataset
from agents import AgentNoPlanningBayesian, AgentDummy
from agent_utils import switch_proposal_perspective

import copy
import random
random.seed(62)

def break_conversation(conversation):
    return conversation[:4], conversation[4:]

def agent_negotiation(agent_tuple, conversation, act_ag=0, length_limit=20, mode='train'):
    conv_prefix, conv_suffix = break_conversation(conversation)

    conv_length = 0

    ## We are working under the assumption that passive steps won't include markers
    agent_tuple[act_ag].step_passive(None, conv_prefix[0], mode=mode)
    act_ag = (act_ag+1)%2
    conv_length += 1

    for dia, dia_next in zip(conv_prefix, conv_prefix[1:]):
        agent_tuple[act_ag].step_passive(switch_proposal_perspective(dia), dia_next, mode=mode)
        act_ag = (act_ag+1)%2
        conv_length += 1

    prev_dialog = conv_prefix[-1]
    while True:
        if(conv_length > length_limit):
            break
        out_dialog = agent_tuple[act_ag].step_active(switch_proposal_perspective(prev_dialog), mode=mode)
        if(out_dialog['text']=='Accept-Deal' or out_dialog['text']=='Walk-Away'):
            break
        conv_length += 1
        act_ag = (act_ag+1)%2
        prev_dialog = out_dialog

    ## Reward setup is required
    return

def train(agent1, agent2):
    ## Load Dataset
    all_data = get_dataset('../casino_with_emotions.json', shuffle=True)

    ## Go through all the dialogues
    for i, (conversation, participant_info) in enumerate(all_data):
        ## Reset Agent State if required at the start of conversation
        agent1.start_conversation()
        agent2.start_conversation()

        ## Setup Agent's priority
        ## Assumption here that the first person doesn't speak twice in a row at the start
        agent1_id = conversation[0]['speaker_id']
        agent1.set_priority(participant_info[agent1_id]['value2issue'])
        agent1.set_name(agent1_id)

        agent2_id = conversation[1]['speaker_id']
        agent2.set_priority(participant_info[agent2_id]['value2issue'])
        agent2.set_name(agent2_id)

        ## Certain agents require complete conversation for training
        agent1.set_conversation(conversation)
        agent2.set_conversation(conversation)

        agent_negotiation([agent1, agent2], copy.deepcopy(conversation), act_ag=0, mode='train', length_limit=20)
        ## Repeat the negotiation again but with reversed roles
        agent_negotiation([agent1, agent2], copy.deepcopy(conversation), act_ag=1, mode='train', length_limit=20)

    ## save file
    agent1.save_model()
    agent2.save_model()

if __name__ == "__main__":
    score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}
    length_penalty = 0.5
    agent1 = AgentDummy(score_weightage, length_penalty, 0)
    agent2 = AgentNoPlanningBayesian(score_weightage, length_penalty, 0)
    train(agent1, agent2)
