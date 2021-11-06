from tqdm import tqdm 
import csv


for split in ['train','validation','test']:
    dataset = "emotion"

    in_dial = open(f"{split}/dialogues_{split}.txt", 'r')
    if ("emotion" in dataset):
        idx2class = ["no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        in_lable = open(f"{split}/dialogues_emotion_{split}.txt", 'r')
    elif("act" in dataset):
        idx2class = ["inform", "question", "directive", "commissive"]
        in_lable = open(f"data/{split}/dialogues_act_{split}.txt", 'r')


    x = []
    y = []
    for i, (line_dial, line_lable) in enumerate(tqdm(zip(in_dial,in_lable), ascii=True)):
        history = line_dial.split('__eou__')
        history = history[:-1]
        history = [h.strip().replace(" , ",", ")
                        .replace(" . ",". ").replace(" .",".")
                        .replace(" ? ","? ").replace(" ?","?")
                        .replace(" ’ ","’").replace(" : ",": ")
                        for h in history]

        lables = line_lable.split(" ")
        lables = lables[:-1]
        if len(lables) != len(history):
            continue
        for id_turn, h in enumerate(history):
            x.append(h)
            if("act" in dataset):
                y.append(int(lables[id_turn])-1)
            else:
                y.append(int(lables[id_turn]))
                


    file_save = f"{split}.tsv"
    with open(file_save, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["text", "label"])
        for x_, y_ in zip(x,y):
            tsv_writer.writerow([x_, y_])
    