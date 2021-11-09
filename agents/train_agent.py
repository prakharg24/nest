from dataloader import get_dataset
from agents import RandomAgentStubborn, RandomAgentConsiderate

def train(agent):
    ## Load Dataset
    all_data = get_dataset('../casino_with_emotions.json')
    print(all_data[0])

    ## Go through all the dialogues
    for conversation in all_data:
        ## Setup Agent's priority

        ## create a conversation prefix if required
        for dialogue in conversation_prefix:
            agent.step_passive(mode='train')

        for dialogue in conversation_suffix:
            ## Check for the correct id
            agent.step(mode='train')

    ## save file
    agent.save_model(outfile)

if __name__ == "__main__":
    agent = RandomAgentStubborn()
    train(agent)
