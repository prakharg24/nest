import json
import copy
import random

extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
label_emotion = ["anger", "fear", "joy" ,"love", "sadness", "surprise"]
# label_emotion = {4: "sadness", 2: "joy", 0: "anger", 1: "fear", 5: "surprise", 3: "love"}
label_intent = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']
emotion_revdict = {ele: i for i, ele in enumerate(label_emotion)}
intent_revdict = {ele: i for i, ele in enumerate(label_intent)}
num_emotion = len(label_emotion)
num_intent = len(label_intent)

def emotion_label_to_index(label):
    return emotion_revdict[label]

def intent_label_to_index(labels):
    empty_array = [0 for _ in range(num_intent)]
    if labels=='':
        return empty_array
    labels = labels.split(",")
    for lbl in labels:
        empty_array[intent_revdict[lbl]] = 1
    return empty_array

def local_proposal_dict(proposal_array):
    outdict = {}
    for ele in proposal_array:
        outdict[ele[0]] = {'Firewood': int(ele[1][0]), 'Water': int(ele[1][1]), 'Food': int(ele[1][2])}
    return outdict

def local_emotion_dict(emotion_array):
    outdict = {}
    for ele in emotion_array:
        outdict[ele[0]] = emotion_label_to_index(ele[1])
    return outdict

def local_intent_dict(intent_array):
    outdict = {}
    for ele in intent_array:
        outdict[ele[0]] = intent_label_to_index(ele[1])
    return outdict

def get_dataset(fname, shuffle=False):
    conversations = []

    data = json.load(open(fname))
    for ind, item in enumerate(data):
        if 'proposals' not in item:
            continue
        if len(item['annotations'])==0:
            continue

        dialogues = []
        proposal_dict = local_proposal_dict(item['proposals'])
        emotion_dict = local_emotion_dict(item['emotions'])
        intent_dict = local_intent_dict(item['annotations'])

        complete_log = item['chat_logs']
        participant_info = item['participant_info']
        prev_speaker = None
        for i, utterance in enumerate(complete_log):
            ## Assumption, the dialogue loader will never account for markers like submit-deal, reject-deal, accept-deal etc.
            dialogue_dict = {}
            dialogue_dict['speaker_id'] = utterance['id']
            dialogue_dict['text'] = utterance['text']
            if utterance['text'] in extra_utterances:
                dialogue_dict['is_marker'] = True
                dialogue_dict['emotion'] = None
                dialogue_dict['intent'] = None
                if utterance['text']=='Submit-Deal':
                    dialogue_dict['proposal'] = {k: int(utterance['task_data']['issue2youget'][k]) for k in utterance['task_data']['issue2youget']}
                else:
                    dialogue_dict['proposal'] = None
            else:
                dialogue_dict['is_marker'] = False
                dialogue_dict['emotion'] = emotion_dict[utterance['text']]
                dialogue_dict['intent'] = intent_dict[utterance['text']]
                dialogue_dict['proposal'] = proposal_dict[utterance['text']]

            ###### NEED TO CLEAN UP CONVERSATIONS WITH CONSECUTIVE SPEAKERS!!!

            dialogues.append(dialogue_dict)

        conversations.append((dialogues, participant_info))

    if shuffle:
        random.shuffle(conversations)
    return conversations
