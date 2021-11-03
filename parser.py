from bertclassifier import get_emotion_label, get_intent_label

class Parser():
    def __init__(max_len=256):
        self.max_len = max_len

    def parse(utterance):
        emotion_label = get_emotion_labeln(utterance)
        print(emotion_label)
        intent_label = get_intent_label(utterance)
        print(intent_label)
        exit()

        return emotion_label, intent_label
