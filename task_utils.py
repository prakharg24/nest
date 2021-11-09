import random
import json
from parser import Parser

def get_proposal_score(priorities, proposal, score_weightage):
    final_score = 0
    for ele in proposal:
        final_score += score_weightage[priorities[ele]]*proposal[ele]

    return final_score

def get_random_emotion():
    return random.randint(0, 5)

def get_random_intent():
    return [random.randint(0, 1) for _ in range(10)]

class NegotiationStarter():
    def __init__(self, datafile):
        self.parser = Parser(debug_mode=True)
        full_conversations = self.load_dialogues_json(datafile)
        print(len(full_conversations))
        exit()
        self.conversations = []
        for ele in full_conversations:
            prefix_ele = self.cut_conversation_prefix(ele)
            if prefix_ele is not None:
                self.conversations.append(prefix_ele)


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
                conversations.append(dialogues)

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
            return conversation[:end_index]

    def get_all_agents(self, conversation):
        agent_counter = 0
        agent_dict = {}
        for ele in conversation:
            if ele['speaker_id'] not in agent_dict:
                agent_dict[ele['speaker_id']] = agent_counter
                agent_counter += 1

        return agent_dict

    def get_random_negotiation_prefix():
        conversation = random.choice(self.conversations)
        agents = self.get_all_agents(conversation)

        return conversation, agents
