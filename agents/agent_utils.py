import random
import json
# from parser import Parser

def get_proposal_score(priorities, proposal, score_weightage):
    final_score = 0
    for ele in proposal:
        final_score += score_weightage[priorities[ele]]*proposal[ele]

    return final_score

def incomplete_proposal(proposal):
    for ele in proposal:
        if proposal[ele]==-1:
            return True
    return False

def switch_proposal_perspective(proposal):
    for ele in proposal:
        if (proposal[ele]!=-1):
            proposal[ele] = 3 - proposal[ele]
    return proposal

def get_random_emotion():
    return random.randint(0, 5)

def get_random_intent():
    return [random.randint(0, 1) for _ in range(10)]

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
