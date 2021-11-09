from bertclassifier.inference import initialize_models, get_emotion_label, get_intent_label

class Parser():
    def __init__(self, max_len=256, debug_mode=False):
        # super(Parser, self).__init__()
        self.max_len = max_len
        self.debug_mode = debug_mode
        if not self.debug_mode:
            self.load_models()

    def load_models(self):
        initialize_models()

    def parse(self, utterance):
        if not self.debug_mode:
            emotion_label, emotion_index, emotion_logits = get_emotion_label(utterance)
            intent_labels, intent_indices, intent_logits = get_intent_label(utterance)

            emotion_dict = {'label' : emotion_label, 'index' : emotion_index, 'logits' : emotion_logits}
            intent_dict = {'label' : intent_labels, 'index' : intent_indices, 'logits' : intent_logits}

            return {'emotion': emotion_dict, 'intent': intent_dict}
        else:
            return {'emotion': None, 'intent': None}
