import random
import json

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
        self.conversations = self.load_dialogues_json(datafile)

    def load_dialogues_json(fname):
        data = json.load(open(fname))
        for ind, item in enumerate(data):
            if (ind < start or ind > end):
                continue
            if (item['dialogue_id'] in weird_cases):
                continue
            if 'proposals' in item:
                continue
            item['proposals'] = []
            print("\n\n Dialogue Number %d Starts \n\n" % (ind+1))
            complete_log = item['chat_logs']

            for i, utterance in enumerate(complete_log):
                if utterance['text'] in extra_utterances:
                    print("Dialogue ID : %d \t Utterance Number : %d" % (item['dialogue_id'], i))
                    if utterance['text'] == 'Submit-Deal':
                        print(utterance['id'], ":", utterance['text'], ":", utterance['task_data']['issue2youget'])
                    else:
                        print(utterance['id'], ":", utterance['text'])
                    continue
                print("Dialogue ID : %d \t Utterance Number : %d" % (item['dialogue_id'], i))
                print(utterance['id'], ":", utterance['text'])
                firewood = input("Firewood : ")
                water = input("Water : ")
                food = input("Food : ")
                speakergets = [firewood, water, food]
                item['proposals'].append([utterance['text'], speakergets])

            with open(outfile, 'w') as f:
                json.dump(data, f)

    def get_random_negotiation_prefix():
