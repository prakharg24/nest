import json
import sys
import os
from shutil import copyfile

outfile = 'casino_with_emotions_and_intents_and_proposals.json'
fname = 'casino_with_emotions_and_intents.json'
proposal_label_file_one = 'binitha_labels.json'
proposal_label_file_two = 'gnanu_labels.json'
proposal_label_file_three = 'prakhar_labels.json'

if not os.path.exists(outfile):
  copyfile(fname, outfile)

data = json.load(open(outfile))
label_one = json.load(open(proposal_label_file_one))
label_two = json.load(open(proposal_label_file_two))
label_three = json.load(open(proposal_label_file_three))

for ind, item in enumerate(data):
    if 'proposals' in item:
        continue
    if ind <= 300:
      item_from_proposal_labelled_file = label_three[ind]
      if 'proposals' in item_from_proposal_labelled_file:
        item['proposals'] = item_from_proposal_labelled_file['proposals']
    elif ind >300 and ind <= 600:
      item_from_proposal_labelled_file = label_two[ind]
      if 'proposals' in item_from_proposal_labelled_file:
        item['proposals'] = item_from_proposal_labelled_file['proposals']
    elif ind <= 900:
      item_from_proposal_labelled_file = label_one[ind]
      if 'proposals' in item_from_proposal_labelled_file:
        item['proposals'] = item_from_proposal_labelled_file['proposals']
    
    with open(outfile, 'w') as f:
      json.dump(data, f)



