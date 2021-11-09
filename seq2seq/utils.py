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
    # Returns dialogue utterances with agent ids and the maximum dialogue length for the file 
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
                agent_id = utterance['id']
                X.append({'text':utterance['text'],'agent_id':agent_id})
                num_of_utterances+=1
        if num_of_utterances > max_length:
          max_length = num_of_utterances
        if len(X) != 0:  
          dialogue_utterances[ item['dialogue_id'] ] = X

    return dialogue_utterances, max_length

def get_one_hot_encoding(utterance_labels):
  # Returns one hot encoded vectors for the labels
  ohv = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for label in utterance_labels:
    if type(label['label']) is str:
      ohv[label_to_index[label['label']]] = 1
    else :
      for lbl in label['label']:
        ohv[label_to_index[lbl]] = 1
  return ohv

def get_src_trg_by_fname(fname):
  # Returns source and target dialogue sequences
  # source sequences --> starting agent's utterance labels
  # target sequences --> other agent's utterance labels

  dialogue_utterances, max_dialogue_length = get_dialogs_from_json(fname)
  ohv_pad = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  print('Creating one hot representation of labels')
  src = [] # source label sequences for all dialogues
  trg = [] # target label sequences for all dialogues
  for dialogue_id, utterances_with_id in dialogue_utterances.items():
    src_per_dialogue = []
    trg_per_dialogue = []
    prev_agent = ""
    start_agent = ""
    i=1
    for utterance_with_agent_id in utterances_with_id:
      
      utterance = utterance_with_agent_id['text']
      agent_id = utterance_with_agent_id['agent_id']
      
      if start_agent == "":
        start_agent = agent_id
      if agent_id == prev_agent:
        print('Same agent here!')
        print('dialogue no: ', dialogue_id)
        # if previous agent and current agent are the same, combine the previous and current utterances to extract labels
        label = parse(prev_utterance + utterance)
      else :
        label = parse(utterance)
      utterance_labels = []
      utterance_labels.append(label['emotion'])
      utterance_labels.append(label['intent'])
      ohv = get_one_hot_encoding(utterance_labels)
      if agent_id == prev_agent:
        # if previous agent and current agent are the same, replace previous ohv with updated ohv
        if agent_id == start_agent:
          src_per_dialogue[-1] = ohv
        else:
          trg_per_dialogue[-1] = ohv
      else:
        if agent_id == start_agent:
          src_per_dialogue.append(ohv)
        else:
          trg_per_dialogue.append(ohv)
      prev_agent = agent_id
      prev_utterance = utterance
      i+=1

    src_per_dialogue = np.array(src_per_dialogue)
    trg_per_dialogue = np.array(trg_per_dialogue)
    
    # if sequence lengths of source and target are not equal, pop the last element of the greater length array (i.e. for every source there should be a response) 
    if src_per_dialogue.shape[0] != trg_per_dialogue.shape[0]:
      if src_per_dialogue.shape[0] > trg_per_dialogue.shape[0]:
        src_per_dialogue = np.delete(src_per_dialogue, -1, 0)
      else:
        trg_per_dialogue = np.delete(trg_per_dialogue, -1, 0)

    src.append(src_per_dialogue)
    trg.append(trg_per_dialogue)
  
  return src, trg

