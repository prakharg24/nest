### Import all components.
### There should a simple python file for each component which contains all required functions and a link to the component folder.
### Check structure.md for details
### Example,
### from ppcm import generate_dialogue
### from emotion_classifier import tokeniser, classification, scorer
### etc. etc.

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
    meta_data = dummy_parser(conv_array[-1]['text'])
    meta_data_seq.append(meta_data)

    ### Call the agent with the new set of metadata
    output_meta_data, output_proposal = dummy_manager(meta_data_seq)
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

    ### Ask for the Human's Input
    user_inp = input("Human : ")

    ### Check if the human input was a proposal and ask for extra inputs if required

    ### Check again if the negotiation has ended
