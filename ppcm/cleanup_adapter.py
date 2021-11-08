import jsonlines
import tqdm
from metric.lm_score import get_ppl
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

clean_dialogues = []
with jsonlines.open(input_file) as reader:
    for i, obj in enumerate(tqdm(reader)):
        if(i>10):
            break
        text = " ".join(tokenize.sent_tokenize(obj["hyp"]["PPLM"][0][-1]))
        starter = copy.deepcopy(obj['conversation']['conversation'])
        score = get_ppl(text, starter)
        if score>700:
            continue
        clean_dialogues.append(obj)

with jsonlines.open(name) as writer:
    for obj in clean_dialogues:
        writer.write(obj)
