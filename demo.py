### Import all components. Check structure.md for details.
### There should a simple python file for each component which contains all required functions and a link to the component folder.
### Example,
### from ppcm import generate_dialogue
### from emotion_classifier import tokeniser, classification, scorer
### from manager import SomeManagerClass
### etc. etc.
import random
import numpy as np
import torch

from parser import Parser

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

### Initialize all hyperparameters and class objects as required
ttl_firewood = 3
ttl_water = 3
ttl_food = 3

# Randomize the priority of elements for both human and AI
elements_to_divide = ["Firewood", "Water", "Food"]
priorities = ["low", "medium", "high"]

np.random.shuffle(elements_to_divide)
human_priority = {e:p for e, p in zip(elements_to_divide, priorities)}

np.random.shuffle(elements_to_divide)
agent_priority = {e:p for e, p in zip(elements_to_divide, priorities)}


# Load manager class according to its priority
dummy_manager = SomeManagerClass(agent_priority)
multi_parser = Parser()

# Show the user their priority
print("Your Priority has been chosen as follows - Firewood : %s, Water : %s, Food : %s" % (human_priority['Firewood'],
                                                                                           human_priority['Water'],
                                                                                           human_priority['Food']))

### Start with some conversation
### We need a starting point of conversation. Let us use some example from dataset and hardcode it for now
print("Conversation Starts")

conv_array = []
conv_array.append({'text': 'Hello there! Are you getting excited for your upcoming trip?! I am so very excited to test my skills!',
                   'id': 'human', 'task_data': {}})

print("Human : %s" % conv_array[0]['text'])

meta_data_seq = []
while True:
    ### Call the parser (or multiple different parsers as required) to extract relevant meta-data
    meta_data = multi_parser.parse(conv_array[-1]['text'])
    meta_data_seq.append(meta_data)

    ### Call the agent with the new set of metadata
    output_meta_data, output_proposal = dummy_manager.step(meta_data_seq)
    meta_data_seq.append(output_meta_data)

    ### Check if the negotiation has moved to proposal stage
    if output_proposal is not None:
        conv_array.append({'text': output_proposal['proposal_phrase'], 'id': 'agent', 'task_data': output_proposal[task_data]})
        print("Agent : %s" % conv_array[-1]['text'])
        print("Firewood : %d" % conv_array[-1]['task_data']['Firewood'])
        print("Water : %d" % conv_array[-1]['task_data']['Water'])
        print("Food : %d" % conv_array[-1]['task_data']['Food'])
    else:
        ### Call the generator to produce the output dialogue for the agent
        out_dialogue = dummy_generator(conv_array, output_meta_data)
        conv_array.append({'text': out_dialogue['text'], 'id': 'agent', 'task_data': {}})
        print("Agent : %s" % conv_array[-1]['text'])

    ### Check if the negotiation has ended
    if conv_array[-1]['text'] == 'Accept-Deal':
        agent_score, human_score = calculate_scores(conv_array[-2]['id'], conv_array[-2]['task_data'])
        print("End of Negotiation")
        print("Agent Score : %d, Human Score : %d" % (agent_score, human_score))
        break

    ### Ask for the Human's Input
    user_inp = input("Human : ")

    ### Check if the human input was a proposal and ask for extra inputs if required
    if user_inp == 'Accept-Deal':
        agent_score, human_score = calculate_scores(conv_array[-1]['id'], conv_array[-1]['task_data'])
        print("End of Negotiation")
        print("Agent Score : %d, Human Score : %d" % (agent_score, human_score))
        break
    elif user_inp == 'Submit-Deal':
        firewood = input("Firewood : ")
        water = input("Water : ")
        food = input("Food : ")
        conv_array.append({'text': user_inp, 'id': 'human', 'task_data': {'Firewood': int(firewood), 'Water': int(water), 'Food': int(food)}})
    elif user_inp == 'Reject-Deal':
        # Ask for any other input text to reason for rejection
        additional_inp = input("Human : ")
        conv_array.append({'text': user_inp + ". " + additional_inp, 'id': 'human', 'task_data': {}})
    else:
        conv_array.append({'text': user_inp, 'id': 'human', 'task_data': {}})
