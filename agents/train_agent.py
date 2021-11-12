from dataloader import get_dataset
from agents import AgentNoPlanningBayesian, AgentDummy, AgentNoPlanningImitation
from agent_utils import switch_proposal_perspective, get_proposal_score

import copy
import random
random.seed(62)

score_weightage = {"High" : 5, "Medium" : 4, "Low" : 3}
length_penalty = 0.5

def break_conversation(conversation):
    return conversation[:4], conversation[4:]

def agent_negotiation(agent_tuple, conversation, participant_info, act_ag=0, length_limit=20):
    record_conversation = []

    ## Reset Agent State if required at the start of conversation
    agent_tuple[act_ag].start_conversation()
    agent_tuple[(act_ag+1)%2].start_conversation()

    ## Setup Agent's priority
    ## Assumption here that the first person doesn't speak twice in a row at the start
    agent1_id = conversation[0]['speaker_id']
    agent_tuple[act_ag].set_priority(participant_info[agent1_id]['value2issue'])
    agent_tuple[act_ag].set_name(agent1_id)

    agent2_id = conversation[1]['speaker_id']
    agent_tuple[(act_ag+1)%2].set_priority(participant_info[agent2_id]['value2issue'])
    agent_tuple[(act_ag+1)%2].set_name(agent2_id)

    ## Certain agents require complete conversation for training
    agent_tuple[act_ag].set_conversation(conversation)
    agent_tuple[(act_ag+1)%2].set_conversation(conversation)

    conv_prefix, conv_suffix = break_conversation(conversation)

    conv_length = 0

    ## We are working under the assumption that passive steps won't include markers
    record_conversation.append(conv_prefix[0])
    agent_tuple[act_ag].step_passive(None, conv_prefix[0])
    act_ag = (act_ag+1)%2
    conv_length += 1

    for dia, dia_next in zip(conv_prefix, conv_prefix[1:]):
        record_conversation.append(dia_next)
        agent_tuple[act_ag].step_passive(switch_proposal_perspective(dia), dia_next)
        act_ag = (act_ag+1)%2
        conv_length += 1

    prev_dialog = conv_prefix[-1]
    while True:
        if(conv_length > length_limit):
            break
        out_dialog = agent_tuple[act_ag].step_active(switch_proposal_perspective(prev_dialog))
        record_conversation.append(out_dialog)
        if(out_dialog['text']=='Accept-Deal' or out_dialog['text']=='Walk-Away'):
            break
        conv_length += 1
        act_ag = (act_ag+1)%2
        prev_dialog = out_dialog

    # print(record_conversation)
    reward_tuple = [0, 0]
    if out_dialog['text']=='Accept-Deal' and prev_dialog['proposal'] is not None:
        conv_length_penalty = length_penalty*conv_length

        reward_tuple[(act_ag+1)%2] = get_proposal_score(agent_tuple[(act_ag+1)%2].priorities, prev_dialog['proposal'], score_weightage) - conv_length_penalty

        prev_dialog_reverted = switch_proposal_perspective(prev_dialog)
        reward_tuple[act_ag] = get_proposal_score(agent_tuple[act_ag].priorities, prev_dialog_reverted['proposal'], score_weightage) - conv_length_penalty

    # if(reward_tuple[0]!=reward_tuple[1]):
    agent_tuple[0].step_reward(reward_tuple[0])
    agent_tuple[1].step_reward(reward_tuple[1])

    return reward_tuple

def train(agent1, agent2):
    ## Load Dataset
    all_data = get_dataset('../casino_with_emotions_and_intents.json', shuffle=True)

    rewards = [0, 0]
    ## Go through all the dialogues
    for i, (conversation, participant_info) in enumerate(all_data):
        reward_tuple = agent_negotiation([agent1, agent2], copy.deepcopy(conversation), participant_info, act_ag=0, length_limit=20)
        rewards[0] += reward_tuple[0]
        rewards[1] += reward_tuple[1]
        reward_tuple = agent_negotiation([agent1, agent2], copy.deepcopy(conversation), participant_info, act_ag=1, length_limit=20)
        rewards[0] += reward_tuple[0]
        rewards[1] += reward_tuple[1]
        print("Rewards Agent 1", rewards[0])
        print("Rewards Agent 2", rewards[1])
        # exit()

    print("Conversation Length : ", len(all_data))
    ## save file
    if agent1.mode=='train':
        agent1.save_model()
    if agent2.mode=='train':
        agent2.save_model()

if __name__ == "__main__":
    training_setup = 2
    if training_setup==1:
        agent1 = AgentDummy(score_weightage, length_penalty, 0)
        agent1.set_mode('eval')
        agent2 = AgentNoPlanningBayesian(score_weightage, length_penalty, 0)
        agent2.load_model()
        agent2.set_mode('eval')
        # agent2.set_mode('train')
        train(agent1, agent2)
    if training_setup==2:
        agent1 = AgentDummy(score_weightage, length_penalty, 0)
        agent1.set_mode('eval')
        agent2 = AgentNoPlanningImitation(score_weightage, length_penalty, 0)
        agent2.load_model()
        agent2.set_mode('eval')
        # agent2.set_mode('train')
        train(agent1, agent2)
