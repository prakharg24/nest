import random
import json
import numpy as np
import copy
import math
# from parser import Parser

def index_to_onehot(index, size):
    outarr = [0 for _ in range(size)]
    outarr[index] = 1

    return outarr

def normalize_prob(prob_arr):
    if np.sum(prob_arr)!=0:
        return prob_arr/np.sum(prob_arr)

    return prob_arr

def choose_random_with_prob(arr, prob):
    if np.sum(prob)==1:
        return np.random.choice(arr, p=prob)
    else:
        return np.random.choice(arr)

def get_proposal_score(priorities, proposal, score_weightage):
    final_score = 0
    for ele in priorities:
        final_score += score_weightage[ele]*proposal[priorities[ele]]

    return final_score

def incomplete_proposal(proposal):
    for ele in proposal:
        if proposal[ele]==-1:
            return True
    return False

def convert_proposal_to_arr(proposal, priorities):
    return [proposal[priorities["High"]], proposal[priorities["Medium"]], proposal[priorities["Low"]]]

def switch_proposal_perspective(inpdict):
    outdict = copy.deepcopy(inpdict)
    if outdict is None:
        return outdict
    if outdict['proposal'] is None:
        return outdict
    for ele in outdict['proposal']:
        if (outdict['proposal'][ele]!=-1):
            outdict['proposal'][ele] = 3 - outdict['proposal'][ele]

    return outdict

def uct_score(utility_value, exploration_term, state_visit_count, visits_count):
    return utility_value + exploration_term*math.sqrt(math.log(state_visit_count+1)/(visits_count+1))

def get_random_emotion():
    return random.randint(0, 5)

def get_random_intent():
    return [random.randint(0, 1) for _ in range(10)]

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def break_conversation(conversation):
    return conversation[:4], conversation[4:]

def agent_negotiation(agent_tuple, conversation, participant_info, length_penalty, score_weightage, act_ag=0, length_limit=20):

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

class NegotiationStarter():
    def __init__(self, datafile):
        self.parser = Parser(debug_mode=True)
        full_conversations = self.load_dialogues_json(datafile)
        self.conversations = []
        for ele in full_conversations:
            prefix_conv = self.cut_conversation_prefix(ele[0])
            if prefix_conv is not None:
                self.conversations.append((prefix_conv, ele[1]))

    def load_dialogues_json(self, fname):
        conversations = []

        data = json.load(open(fname))
        extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
        for ind, item in enumerate(data):
            if 'proposals' not in item:
                continue

            dialogues = []
            proposal_dict = self.local_proposal_dict(item['proposals'])
            complete_log = item['chat_logs']
            participant_info = item['participant_info']
            for i, utterance in enumerate(complete_log):
                ## Assumption, the dialogue loader will never account for markers like submit-deal, reject-deal, accept-deal etc.
                if utterance['text'] in extra_utterances:
                    break

                parse_data = self.parser.parse(utterance['text'])
                proposal_data = proposal_dict[utterance['text']]

                dialogues.append({'speaker_id': utterance['id'],
                                  'emotion': parse_data['emotion'],
                                  'intent': parse_data['intent'],
                                  'proposal': proposal_data})

            if len(dialogues) > 5:
                conversations.append((dialogues, participant_info))

        return conversations

    def local_proposal_dict(self, proposal_array):
        outdict = {}
        for ele in proposal_array:
            outdict[ele[0]] = {'Firewood': int(ele[1][0]), 'Water': int(ele[1][1]), 'Food': int(ele[1][2])}

        return outdict

    def cut_conversation_prefix(self, conversation):
        end_index = -1
        for ite, dialogue in enumerate(conversation):
            ## Check if some preference has been elicited
            if (dialogue['proposal']['Firewood']!=-1 or dialogue['proposal']['Water']!=-1 or dialogue['proposal']['Food']!=-1):
                end_index = ite
                break

        if end_index==-1:
            return None
        elif end_index < 2:
            return conversation[:2]
        else:
            return conversation[:(end_index+1)]

    def get_all_agents_and_priorities(self, conversation):
        agent_dict = {}
        for ele in conversation[0]:
            if ele['speaker_id'] not in agent_dict:
                participant_info = conversation[1][ele['speaker_id']]['value2issue']
                agent_dict[ele['speaker_id']] = {participant_info[k]:k for k in participant_info}

        return agent_dict

    def get_random_negotiation_prefix(self):
        conversation = random.choice(self.conversations)
        agents = self.get_all_agents_and_priorities(conversation)

        return conversation[0], agents
