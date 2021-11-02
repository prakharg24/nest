from bertclassifier import emotion_extraction, intent_extraction

class Parser():
    def __init__(max_len=256):
        self.max_len = max_len

    def parse(utterance):
        emotion_label = emotion_extraction(utterance)
        intent_label = intent_extraction(utterance)
