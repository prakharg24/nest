from bertclassifier.inference import get_emotion_label, get_intent_label
import json
import numpy as np

label_to_index = {'elicit-pref':0, 'no-need':1, 'uv-part':2, 'other-need':3, 'showing-empathy':4, 'vouch-fair':5, 'small-talk':6, 'self-need':7, 'promote-coordination':8, 'non-strategic':9, "sadness": 10, "joy": 11, "anger":12, "fear":13, "surprise":14, "love":15}

def parse(utterance):
        emotion_label, emotion_index, emotion_logits = get_emotion_label(utterance)
        intent_labels, intent_indices, intent_logits = get_intent_label(utterance)

        emotion_dict = {'label' : emotion_label, 'index' : emotion_index, 'logits' : emotion_logits}
        intent_dict = {'label' : intent_labels, 'index' : intent_indices, 'logits' : intent_logits}

        return {'emotion': emotion_dict, 'intent': intent_dict}


def make_anno_dict(anno_arr):
    outdict = {}

    for ele in anno_arr:
        outdict[ele[0]] = ele[1]

    return outdict

def get_dialogs_from_json(fname):
    print('Loading dialogues from file')
    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
    annotation_list = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']
    max_length = 0

    data = json.load(open(fname))

    dialogue_utterances = {}

    for item in data:
        X = []
        num_of_utterances = 0
        complete_log = item['chat_logs']
        annotations = make_anno_dict(item['annotations'])

        for i, utterance in enumerate(complete_log):
            if utterance['text'] in extra_utterances:
                continue
            elif utterance['text'] in annotations:
                X.append((utterance['text']))
                num_of_utterances+=1
        if num_of_utterances > max_length:
          max_length = num_of_utterances  
        dialogue_utterances[ item['dialogue_id'] ] = X

    return dialogue_utterances, max_length

def get_one_hot_encoding(utterance_labels):
  ohv = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for label in utterance_labels:
    if type(label['label']) is str:
      ohv[label_to_index[label['label']]] = 1
    else :
      for lbl in label['label']:
        ohv[label_to_index[lbl]] = 1
  return ohv

def get_src_trg_by_fname(fname):
  dialogue_utterances, max_dialogue_length = get_dialogs_from_json(fname)
  ohv_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  print('Creating one hot representation of labels')
  src = []
  trg = []
  for dialogue_id, utterances in dialogue_utterances.items():
    src_per_dialogue = []
    trg_per_dialogue = []
    ohv_utterances_in_dialogue = []
    for utterance in utterances:
      label = parse(utterance)
      utterance_labels = []
      utterance_labels.append(label['emotion'])
      utterance_labels.append(label['intent'])
      ohv = get_one_hot_encoding(utterance_labels)
      ohv_utterances_in_dialogue.append(ohv)
    if len(ohv_utterances_in_dialogue) < max_dialogue_length:
      diff = max_dialogue_length - len(ohv_utterances_in_dialogue)
      for i in range(0, diff):
        ohv_utterances_in_dialogue.append(ohv_pad)
    src_per_dialogue = ohv_utterances_in_dialogue
    trg_per_dialogue = ohv_utterances_in_dialogue[1:]
    trg_per_dialogue.append( ohv_pad )
    src.append(src_per_dialogue)
    trg.append(trg_per_dialogue)
  src = np.array(src)
  # src = np.transpose(src, (1, 0, 2))
  trg = np.array(trg)
  # trg = np.transpose(trg, (1, 0, 2))
  return src, trg

