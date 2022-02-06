import os
import numpy as np

## For MCTS training ns isolation followed by testing with the same agent
# rel_index = [16, 17]
# model_name = 'MCTS'

## For Q Learning training in isolation followed by testing with the same agent
# rel_index = [22, 23]
# model_name = 'Q Learning'

## For Deep Q Learning training in isolation followed by testing with the same agent
# rel_index = [28, 29]
# model_name = 'Deep Q Learning'

## For MCTS training is isolation followed by testing with a different random seed agent
# rel_index = [46, 47]
# model_name = 'MCTS'

## For Q Learning training is isolation followed by testing with a different random seed agent
# rel_index = [52, 53]
# model_name = 'Q Learning'

## For Deep Q Learning training is isolation followed by testing with a different random seed agent
# rel_index = [58, 59]
# model_name = 'Deep Q Learning'

## For MCTS training is isolation followed by testing with society of isolated agents
# rel_index = [68, 69, 70, 71]
# model_name = 'MCTS'

## For Q Learning training is isolation followed by testing with society of isolated agents
# rel_index = [68, 69, 70, 71]
# model_name = 'Q Learning'

## For Deep Q Learning training is isolation followed by testing with society of isolated agents
# rel_index = [68, 69, 70, 71]
# model_name = 'Deep Q Learning'

## For MCTS training is society followed by testing with same baseline agent only
# rel_index = [112, 113]
# model_name = 'MCTS'

## For Q learning training is society followed by testing with same baseline agent only
# rel_index = [118, 119]
# model_name = 'Q Learning'

## For Deep Q learning training is society followed by testing with same baseline agent only
# rel_index = [124, 125]
# model_name = 'Deep Q Learning'

## For MCTS training is society followed by testing with a different random seed baseline agent only
# rel_index = [142, 143]
# model_name = 'MCTS'

## For Q Learning training is society followed by testing with a different random seed baseline agent only
# rel_index = [148, 149]
# model_name = 'Q Learning'

## For Deep Q Learning training is society followed by testing with a different random seed baseline agent only
# rel_index = [154, 155]
# model_name = 'Deep Q Learning'

## For MCTS training is society followed by testing with society of agents similarly trained but with different random seeds
# rel_index = [164, 165, 166, 167]
# model_name = 'MCTS'

## For Q Learning training is society followed by testing with society of agents similarly trained but with different random seeds
# rel_index = [164, 165, 166, 167]
# model_name = 'Q Learning'

## For Deep Q Learning training is society followed by testing with society of agents similarly trained but with different random seeds
rel_index = [164, 165, 166, 167]
model_name = 'Deep Q Learning'

# parent_dir = 'exp1/dataset/'
# parent_dir = 'exp1/bayesian/'
parent_dir = 'exp1/imitation/'

score_arr = []

for file in os.listdir(parent_dir):
    counter = 0
    with open(os.path.join(parent_dir, file), 'r') as f:
        all_text = f.read().split("\n")
        for line in all_text:
            if len(line)!=0 and line[0]=="|":
                line_split = line.split("|")
                if line_split[1].strip() == 'Agent Type':
                    continue

                if counter in rel_index:
                    if line_split[1].strip() == model_name:
                        score = float(line_split[-2])
                        score_arr.append(score)

                counter += 1

print(score_arr)
print(np.mean(score_arr))
print(np.std(score_arr))
