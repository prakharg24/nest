import json
import sys
import os
from shutil import copyfile

start = int(sys.argv[1])
end = int(sys.argv[2])
outfile = sys.argv[3]
fname = 'bertclassifier/data/casino/casino_train.json'

weird_cases = [648, 681]

if not os.path.exists(outfile):
    copyfile(fname, outfile)

extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']

data = json.load(open(outfile))

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

# return X, Y
