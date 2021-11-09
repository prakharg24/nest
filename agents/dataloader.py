import json
import copy

extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
label_emotion = ["anger", "fear", "joy" ,"love", "sadness", "surprise"]
# label_emotion = {4: "sadness", 2: "joy", 0: "anger", 1: "fear", 5: "surprise", 3: "love"}
label_intent = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']
emotion_revdict = {ele: i for i, ele in enumerate(label_emotion)}
intent_revdict = {ele: i for i, ele in enumerate(label_intent)}

def local_proposal_dict(proposal_array):
    outdict = {}
    for ele in proposal_array:
        outdict[ele[0]] = {'Firewood': int(ele[1][0]), 'Water': int(ele[1][1]), 'Food': int(ele[1][2])}

    return outdict

def local_emotion_dict(emotion_array):
    outdict = {}
    for ele in emotion_array:
        outdict[ele[0]] = emotion_revdict[ele[1]]

    return outdict

def local_intent_dict(intent_array):
    outdict = {}
    empty_array = [0 for _ in range(len(label_intent))]
    for ele in intent_array:
        temp_arr = copy.deepcopy(empty_array)
        labels = ele[1].split(",")
        for label in labels:
            temp_arr[intent_revdict[label]] = 1
        outdict[ele[0]] = temp_arr

    return outdict


def get_dataset(fname):
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
                    dialogue_dict['proposal'] = utterance['task_data']['issue2youget']
                else:
                    dialogue_dict['proposal'] = None
            else:
                dialogue_dict['is_marker'] = False
                dialogue_dict['emotion'] = emotion_dict[utterance['text']]
                dialogue_dict['intent'] = intent_dict[utterance['text']]
                dialogue_dict['proposal'] = proposal_dict[utterance['text']]


            dialogues.append(dialogue_dict)

        conversations.append((dialogues, participant_info))

    return conversations
