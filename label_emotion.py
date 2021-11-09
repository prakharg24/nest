import json
import sys
import os
from shutil import copyfile
from parser import Parser

multi_parser = Parser()

outfile = sys.argv[1]

extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']

data = json.load(open(outfile))

for ind, item in enumerate(data):
    print(ind)
    item['emotions'] = []
    complete_log = item['chat_logs']

    for i, utterance in enumerate(complete_log):
        if utterance['text'] in extra_utterances:
            continue
        parsedict = parser.parse(utterance['text'])
        print(utterance['text'], parsedict)
        exit()
        item['emotions'].append([utterance['text'], parsedict['emotion']['label']])

with open(outfile, 'w') as f:
    json.dump(data, f)

# return X, Y
