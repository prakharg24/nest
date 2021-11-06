import torch
import torch.nn.functional as F
import numpy as np

from models.heads import Discriminator

device = "cuda"

### Sentiment Dataset
sent_idx2class = ["positive", "negative", "very positive", "very negative", "neutral"]
sent_class2idx = {c: i for i, c in enumerate(sent_idx2class)}
sent_activation = F.softmax

### Emotion Dataset
emot_idx2class = ["anger", "fear", "joy", "love", "sadness", "surprise"]
emot_class2idx = {c: i for i, c in enumerate(emot_idx2class)}
emot_activation = F.softmax

### Intent Dataset
intn_idx2class = ['elicit-pref', 'no-need', 'uv-part', 'other-need', 'showing-empathy', 'vouch-fair', 'small-talk', 'self-need', 'promote-coordination', 'non-strategic']
intn_class2idx = {c: i for i, c in enumerate(intn_idx2class)}
intn_activation = torch.sigmoid

example_sentence = "What kind of items do you need?  I especially need water because I would like to do extra hiking and would need to hydrate."
max_length_seq = 128

discriminator = Discriminator(
    class_size=len(sent_idx2class),
    pretrained_model="medium",
    cached_mode=False,
    load_weight='models/discriminators/DIALOGPT_sentiment_classifier_head_best.pt'
).to(device)

# discriminator.get_classifier().load_state_dict(torch.load())

discriminator.eval()

print(example_sentence)
seq = discriminator.tokenizer.encode(example_sentence)
print(seq)
seq = torch.tensor(seq, device=device, dtype=torch.long)
seq.to(device)
seq = seq.unsqueeze(0)
output = discriminator(seq)

# pred = torch.sigmoid(output).cpu().detach().numpy()
pred = output.argmax(dim=1).cpu().detach().numpy().tolist()

print(sent_idx2class[pred[0]])
# indices = [e[0] for e in np.argwhere(pred[0] >=0.5)]
# print([intn_idx2class[ele] for ele in indices])
