import copy
import math
import numpy as np
from scipy.sparse import csr_matrix
import json
import pickle
from agent_utils import get_random_emotion, get_random_intent, choose_random_with_prob, normalize_prob
from agent_utils import get_proposal_score, incomplete_proposal, switch_proposal_perspective, convert_proposal_to_arr
from dataloader import label_emotion, label_intent, num_emotion, num_intent, emotion_label_to_index, intent_label_to_index

### define agents
class AgentTabular():
    def __init__(self, score_weightage, length_penalty, id):
        ## Add all initializations as required
        self.score_weightage = score_weightage
        self.length_penalty = length_penalty
        self.id = id

    def set_priority(self, priorities):
        ## Important Assumption. We sort the priorities as high, medium and low which is relevant for certain parts of the code
        ## Copy it accordingly for any inherited class
        sort_by = ["High", "Medium", "Low"]
        priorities = {k: priorities[k] for k in sort_by}
        self.priorities = priorities

    def set_name(self, agent_name):
        self.name = agent_name

    def set_conversation(self, conversation):
        self.conversation = conversation

    def save_model(self, outfile='some_fixed_file.pt'):
        ## Save the model parameters/dict etc. so that it can be easily laoded
        return

    def load_model(self, infile='some_fixed_file.pt'):
        ## load the model parameters back
        return

    def step_passive(self, input_dict, output_dict, mode='train'):
        ### Skeleton Function. Inherit this and change to memorise the agent's history
        return

    def step_active(self, input_dict, mode='eval'):
        ### Skeleton Function. Inherit this and change to return some sensible proposal
        return input_dict

    def step_reward(self, reward):
        ### Use this function for rewards update at the end of every conversation
        return

    def start_conversation(self):
        ### Use this function to reset any parameters if required before starting a brand new conversation
        return

### Required for training of agents which use actual dataset based calculations
class AgentDummy(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)
        self.conversation_count = 0

    def step_passive(self, input_dict, output_dict, mode='train'):
        return_ind = None
        for ind in range(self.conversation_count, len(self.conversation)):
            if self.conversation[ind]['speaker_id'] == self.name:
                return_ind = ind
                break

        self.conversation_count = return_ind + 1

    def step_active(self, input_dict, mode=None):
        return_ind = None
        for ind in range(self.conversation_count, len(self.conversation)):
            if self.conversation[ind]['speaker_id'] == self.name:
                return_ind = ind
                break

        if return_ind is None:
            ## Walk away from the conversation
            return {'speaker_id' : self.name, 'text' : 'Walk-Away', 'is_marker' : True,
                    'emotion' : None, 'intent' : None, 'proposal' : None}

        self.conversation_count = return_ind + 1
        return self.conversation[ind]

    def start_conversation(self):
        self.conversation_count = 0

### Bayesian Agent that uses existing data to create probability arrays
class AgentNoPlanningBayesian(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)
        self.emotion_trans_count = np.zeros((num_emotion, num_emotion)) ## transition probability between emotions
        self.intent_trans_count = np.zeros((num_intent, 2, 2)) ## transition probability for each intent separately
        self.proposal_prevproposal_joint_count = {"High": np.zeros((4, 4)),
                                                 "Medium": np.zeros((4, 4)),
                                                 "Low": np.zeros((4, 4))} ## joint probability of current and prev proposals
        self.proposal_emotion_joint_count = {"High": np.zeros((4, num_emotion)),
                                            "Medium": np.zeros((4, num_emotion)),
                                            "Low": np.zeros((4, num_emotion))} ## joint probability of proposal and emotion
        self.proposal_intent_joint_count = {"High": np.zeros((4, num_intent, 2)),
                                            "Medium": np.zeros((4, num_intent, 2)),
                                            "Low": np.zeros((4, num_intent, 2))} ## joint probability of proposal and each intent separately

        self.acceptance_count = np.zeros((4, 4, 4)) ## Count All Different Types of Deals Accepted. The ordering of elements is high, medium, low
        self.history = None

    def step_passive(self, input_dict, output_dict, mode='train'):
        if mode != 'train':
            ## No history saving required in the eval mode
            return
        if input_dict is None:
            ## Just the first spoken dialogue. Skip
            return

        if input_dict['is_marker'] or output_dict['is_marker']:
            ## Marker sentences.
            if input_dict['text']=='Submit-Deal' and output_dict['text']=='Accept-Deal':
                ## The submitted deal will not contain a -1
                acceptance_index = convert_proposal_to_arr(input_dict['proposal'], self.priorities)
                self.acceptance_count[tuple(acceptance_index)] += 1
            return

        self.set_emotion_counts(input_dict, output_dict)
        self.set_intent_counts(input_dict, output_dict)
        self.set_proposal_counts(input_dict, output_dict)

    def step_active(self, input_dict, mode='eval'):
        if mode == 'train':
            ## Switch perspective since we are looking for other speaker's utterance
            input_reveresed = switch_proposal_perspective(input_dict)
            for ind, ele in enumerate(self.conversation):
                if ele == input_reveresed:
                    curr_dia = ind
                    break
            self.step_passive(input_dict, self.conversation[curr_dia+1], mode=mode)
            return self.conversation[curr_dia+1]

        if input_dict is None:
            ## The case when the agent needs to speak first
            raise Exception("Input Dict was Empty. The Agents are not trained to actively start negotiations")

        if input_dict['is_marker']:
            ## Marker cases
            if input_dict['text']=='Submit-Deal':
                acceptance_index = convert_proposal_to_arr(input_dict['proposal'], self.priorities)
                acceptance_prob = np.sum(self.acceptance_count[:acceptance_index[0]+1, :acceptance_index[1]+1, :acceptance_index[2]+1])/np.sum(self.acceptance_count)
                is_accepted = np.random.choice([True, False], p=[acceptance_prob, 1-acceptance_prob])

                if is_accepted:
                    return {'speaker_id' : self.name, 'text' : 'Accept-Deal', 'is_marker' : True,
                            'emotion' : None, 'intent' : None, 'proposal' : None}
                else:
                    return {'speaker_id' : self.name, 'text' : 'Reject-Deal', 'is_marker' : True,
                            'emotion' : None, 'intent' : None, 'proposal' : None}

            elif input_dict['text']=='Reject-Deal':
                ## Use the last conversation again to regenerate a proposal
                input_dict = self.history

        ## Calculate P(Emotion | Prev Emotion) using the stored counts and then sample one emotion
        emotion_choice_arr = normalize_prob(self.emotion_trans_count[:, input_dict['emotion']])
        out_emotion = choose_random_with_prob(range(num_emotion), emotion_choice_arr)

        ## Calculate P(Intent | Prev Intent) using the stored counts and then sample one intent
        out_intent = []
        for ite in range(num_intent):
            intent_choice_arr = normalize_prob(self.intent_trans_count[ite, :, input_dict['intent'][ite]])
            out_intent.append(choose_random_with_prob(range(2), intent_choice_arr))

        ## Calculate P(Proposal | Prev Proposal, Emotion, Intent) = P(Emotion | Proposal) * P(Intent | Proposal) * P(Prev Proposal | Proposal) / Z
        ## Z is the normalising factor
        out_proposal_dict = {}
        input_proposal_arr = convert_proposal_to_arr(input_dict['proposal'], self.priorities)
        for ind, priority in enumerate(self.priorities):
            if input_proposal_arr[ind]==-1:
                out_proposal_dict[self.priorities[priority]] = np.random.choice([-1, 3-ind])
                continue

            prob_emotion_given_proposal = normalize_prob(self.proposal_emotion_joint_count[priority][:, out_emotion])
            prob_numerator = prob_emotion_given_proposal
            for ite in range(num_intent):
                prob_intent_given_proposal = normalize_prob(self.proposal_intent_joint_count[priority][:, ite, out_intent[ite]])
                prob_numerator = prob_numerator * prob_intent_given_proposal
            prob_prevproposal_given_proposal = normalize_prob(self.proposal_prevproposal_joint_count[priority][:, input_proposal_arr[ind]])
            prob_numerator = prob_numerator * prob_prevproposal_given_proposal

            proposal_choice_arr = normalize_prob(prob_numerator)
            out_proposal_dict[self.priorities[priority]] = choose_random_with_prob(range(4), proposal_choice_arr)

        self.history = input_dict

        if input_dict['proposal'] == out_proposal_dict and not incomplete_proposal(input_dict['proposal']):
            ## It seems liek something has been agreed upon. Submit a deal
            return {'speaker_id' : self.name, 'text' : 'Submit-Deal', 'is_marker' : True,
                    'emotion' : None, 'intent' : None, 'proposal' : out_proposal_dict}

        return {'speaker_id' : self.name,
                'text' : 'Bayesian Agent does not generate text.',
                'is_marker' : False,
                'emotion' : out_emotion,
                'intent' : out_intent,
                'proposal' : out_proposal_dict}


    def set_emotion_counts(self, input_dict, output_dict):
        self.emotion_trans_count[input_dict['emotion'], output_dict['emotion']] += 1

    def set_intent_counts(self, input_dict, output_dict):
        for counter, (e1, e2) in enumerate(zip(input_dict['intent'], output_dict['intent'])):
            self.intent_trans_count[counter, e1, e2] += 1

    def set_proposal_counts(self, input_dict, output_dict):
        input_arr = convert_proposal_to_arr(input_dict['proposal'], self.priorities)
        output_arr = convert_proposal_to_arr(output_dict['proposal'], self.priorities)

        for ind, priority in enumerate(self.priorities):
            if input_arr[ind]==-1 or output_arr[ind]==-1:
                continue

            self.proposal_prevproposal_joint_count[priority][output_arr[ind], input_arr[ind]] += 1
            self.proposal_emotion_joint_count[priority][output_arr[ind], output_dict['emotion']] += 1
            for counter, e1 in enumerate(output_dict['intent']):
                self.proposal_intent_joint_count[priority][output_arr[ind], counter, e1] += 1

    def save_model(self, outfile='models/bayesian_v1.pkl'):
        outdict = {'emotion_trans_count'                : self.emotion_trans_count,
                   'intent_trans_count'                 : self.intent_trans_count,
                   'proposal_prevproposal_joint_count'  : self.proposal_prevproposal_joint_count,
                   'proposal_emotion_joint_count'       : self.proposal_emotion_joint_count,
                   'proposal_intent_joint_count'        : self.proposal_intent_joint_count,
                   'acceptance_count'                   : self.acceptance_count}

        with open(outfile, 'wb') as fp:
            pickle.dump(outdict, fp)

    def load_model(self, infile='models/bayesian_v1.pkl'):
        with open(infile, 'rb') as fp:
            indict = pickle.load(fp)

        self.emotion_trans_count                = indict['emotion_trans_count']
        self.intent_trans_count                 = indict['intent_trans_count']
        self.proposal_prevproposal_joint_count  = indict['proposal_prevproposal_joint_count']
        self.proposal_emotion_joint_count       = indict['proposal_emotion_joint_count']
        self.proposal_intent_joint_count        = indict['proposal_intent_joint_count']
        self.acceptance_count                   = indict['acceptance_count']

    def start_conversation(self):
        self.history = None
        self.priorities = None
        self.name = None
        self.conversation = None

### MCTS Agent
class AgentMCTS(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)
        state_space = []
        state_space.append(num_emotion) ## For emotions
        state_space.extend([2 for _ in range(num_intent)]) ## For intent
        state_space.extend([4, 4, 4]) ## For proposals
        self.state_space_dim = state_space

        state_space_size = np.prod(self.state_space_dim)
        self.state_visit_counts = csr_matrix((state_space_size, 1))
        self.state_action_visit_counts = csr_matrix((state_space_size, state_space_size))
        self.utility_space = csr_matrix((state_space_size, state_space_size))

        self.marker_state_visit_counts = np.zeros((4, 4, 4))
        self.marker_action_visit_counts = np.zeros((4, 4, 4, 2))
        self.marker_utility_space = np.zeros((4, 4, 4, 2))

        self.trial_visits = []
        self.marker_visits = []
        self.history = None

        self.exploration_term = 1
        self.marker_exploration_term = 1

    def step_passive(self, input_dict, output_dict, mode='train'):
        if mode != 'train':
            ## We don't need to record history is agent is in evaluation mode
            return
        if input_dict is None:
            ## First dialogue with no input. Skip this
            return

        current_state = self.get_state_from_dict(input_dict)
        current_action = self.get_state_from_dict(output_dict)
        self.trial_visits.append((self.state_to_index(current_state), self.state_to_index(current_action)))

    def step_active(self, input_dict, mode='eval'):
        if input_dict is None:
            ## The case when the agent needs to speak first
            raise Exception("Input Dict was Empty. The Agents are not trained to start negotiations")

        if input_dict['is_marker']:
            ## Marker cases
            if input_dict['text']=='Submit-Deal':
                ## How to Mark Acceptance in MCTS?????
                acceptance_arr = convert_proposal_to_arr(input_dict['proposal'], self.priorities)
                acceptance_index = tuple(acceptance_arr)

                marker_utility_arr = self.marker_utility_space[acceptance_index]
                marker_visits_arr = self.marker_action_visit_counts[acceptance_index]
                marker_state_visit_count = self.marker_state_visit_counts[acceptance_index]

                acceptance_score = marker_utility_arr[0] + self.marker_exploration_term*math.sqrt(math.log(marker_state_visit_count+1)/(marker_visits_arr[0]+1))
                rejection_score = marker_utility_arr[1] + self.marker_exploration_term*math.sqrt(math.log(marker_state_visit_count+1)/(marker_visits_arr[1]+1))

                is_accepted = acceptance_score >= rejection_score
                if mode == 'train':
                    self.marker_visits.append((acceptance_index, int(is_accepted)))

                if is_accepted:
                    return {'speaker_id' : self.name, 'text' : 'Accept-Deal', 'is_marker' : True,
                            'emotion' : None, 'intent' : None, 'proposal' : None}
                else:
                    return {'speaker_id' : self.name, 'text' : 'Reject-Deal', 'is_marker' : True,
                            'emotion' : None, 'intent' : None, 'proposal' : None}

            elif input_dict['text']=='Reject-Deal':
                ## Use the last conversation again to regenerate a proposal
                input_dict = self.history

        current_state = self.get_state_from_dict(input_dict)
        current_state_index = self.state_to_index(current_state)

        utility_arr = self.utility_space[current_state_index, :]
        visits_arr = self.state_action_visit_counts[current_state_index, :]
        state_visit_count = self.state_visit_counts[current_state_index, 0]

        indices = visits_arr.nonzero()
        indices = indices[1]

        all_indices = set(range(np.prod(self.state_space_dim)))
        exploration_indices = np.random.choice(list(all_indices - set(indices)), size=10)

        indices = np.concatenate((indices, exploration_indices))

        best_score = -1e10
        best_ind_arr = []
        for ind in indices:
            utility_value = utility_arr[0, ind]
            visits_count = visits_arr[0, ind]
            if mode == 'train':
                score = utility_value + self.exploration_term*math.sqrt(math.log(state_visit_count+1)/(visits_count+1))
            else:
                score = utility_value
            if score == best_score:
                best_ind_arr.append(ind)
            if score > best_score:
                best_score = score
                best_ind_arr = [ind]

        best_action = np.random.choice(best_ind_arr)

        if mode == 'train':
            self.trial_visits.append((current_state_index, best_action))

        output_dict = self.get_dict_from_state(self.index_to_state(best_action))

        self.history = input_dict
        if input_dict['proposal'] == output_dict['proposal'] and not incomplete_proposal(input_dict['proposal']):
            ## It seems like something has been agreed upon. Submit a deal
            return {'speaker_id' : self.name, 'text' : 'Submit-Deal', 'is_marker' : True,
                    'emotion' : None, 'intent' : None, 'proposal' : output_dict['proposal']}

        return output_dict

    def step_reward(self, reward):
        for ele in self.trial_visits:
            self.state_visit_counts[ele[0], 0] += 1
            self.state_action_visit_counts[ele[0], ele[1]] += 1
            self.utility_space[ele[0], ele[1]] += reward

        for ele in self.marker_visits:
            self.marker_state_visit_counts[ele[0], 0] += 1
            self.marker_action_visit_counts[ele[0], ele[1]] += 1
            self.marker_utility_space[ele[0], ele[1]] += reward

    def set_priority(self, priorities):
        sort_by = ["High", "Medium", "Low"]
        priorities = {k: priorities[k] for k in sort_by}
        self.priorities = priorities

    def get_state_from_dict(self, inpdict):
        state = []
        state.append(inpdict['emotion'])
        state.extend(inpdict['intent'])
        state.extend(convert_proposal_to_arr(inpdict['proposal'], self.priorities))

        return state

    def get_dict_from_state(self, state):
        outdict = {}
        outdict['speaker_id'] = self.name
        outdict['text'] = 'MCTS Agent does not generate text.'
        outdict['is_marker'] = False
        outdict['emotion'] = state[0]
        outdict['intent'] = state[1:11]
        proposal = {}
        proposal[self.priorities["High"]] = state[-3]
        proposal[self.priorities["Medium"]] = state[-2]
        proposal[self.priorities["Low"]] = state[-1]
        outdict['proposal'] = proposal

        return outdict

    def state_to_index(self, state):
        index = 0
        mult = np.prod(self.state_space_dim)
        for ele, ref in zip(state, self.state_space_dim):
            mult = mult//ref
            index += ele*mult

        return index

    def index_to_state(self, index):
        state = []
        mult = np.prod(self.state_space_dim)
        for ref in self.state_space_dim:
            mult = mult//ref
            state.append(index//mult)
            index = index%mult

        return state

    def save_model(self, outfile='models/mcts_v1.pkl'):
        outdict = {'state_space_dim'                : self.state_space_dim,
                   'state_visit_counts'             : self.state_visit_counts,
                   'state_action_visit_counts'      : self.state_action_visit_counts,
                   'utility_space'                  : self.utility_space,
                   'marker_state_visit_counts'      : self.marker_state_visit_counts,
                   'marker_action_visit_counts'     : self.marker_action_visit_counts,
                   'marker_utility_space'           : self.marker_utility_space,
                   'exploration_term'               : self.exploration_term,
                   'marker_exploration_term'        : self.marker_exploration_term}

        with open(outfile, 'wb') as fp:
            pickle.dump(outdict, fp)

    def load_model(self, infile='models/mcts_v1.pkl'):
        with open(infile, 'rb') as fp:
            indict = pickle.load(fp)

        self.state_space_dim                = indict['state_space_dim']
        self.state_visit_counts             = indict['state_visit_counts']
        self.state_action_visit_counts      = indict['state_action_visit_counts']
        self.utility_space                  = indict['utility_space']
        self.marker_state_visit_counts      = indict['marker_state_visit_counts']
        self.marker_action_visit_counts     = indict['marker_action_visit_counts']
        self.marker_utility_space           = indict['marker_utility_space']
        self.exploration_term               = indict['exploration_term']
        self.marker_exploration_term        = indict['marker_exploration_term']

    def start_conversation(self):
        self.history = None
        self.priorities = None
        self.name = None
        self.conversation = None
        self.trial_visits = []
        self.marker_visits = []
        return

### MCTS Agent
class AgentQLearning(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        super().__init__(score_weightage, length_penalty, id)
        state_space = []
        state_space.append(num_emotion) ## For emotions
        state_space.extend([2 for _ in range(num_intent)]) ## For intent
        state_space.extend([4, 4, 4]) ## For proposals
        self.state_space_dim = state_space

        state_space_size = np.prod(self.state_space_dim)
        self.state_visit_counts = csr_matrix((state_space_size, 1))
        self.state_action_visit_counts = csr_matrix((state_space_size, state_space_size))
        self.utility_space = csr_matrix((state_space_size, state_space_size))

        self.marker_state_visit_counts = np.zeros((4, 4, 4))
        self.marker_action_visit_counts = np.zeros((4, 4, 4, 2))
        self.marker_utility_space = np.zeros((4, 4, 4, 2))

        self.trial_visits = []
        self.marker_visits = []
        self.history = None

        self.exploration_term = 1
        self.marker_exploration_term = 1

    def step_passive(self, input_dict, output_dict, mode='train'):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        if mode != 'train':
            ## We don't need to record history is agent is in evaluation mode
            return
        if input_dict is None:
            ## First dialogue with no input. Skip this
            return

        current_state = self.get_state_from_dict(input_dict)
        current_action = self.get_state_from_dict(output_dict)
        self.trial_visits.append((self.state_to_index(current_state), self.state_to_index(current_action)))

    def step_active(self, input_dict, mode='eval'):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        if input_dict is None:
            ## The case when the agent needs to speak first
            raise Exception("Input Dict was Empty. The Agents are not trained to start negotiations")

        if input_dict['is_marker']:
            ## Marker cases
            if input_dict['text']=='Submit-Deal':
                ## How to Mark Acceptance in MCTS?????
                acceptance_arr = convert_proposal_to_arr(input_dict['proposal'], self.priorities)
                acceptance_index = tuple(acceptance_arr)

                marker_utility_arr = self.marker_utility_space[acceptance_index]
                marker_visits_arr = self.marker_action_visit_counts[acceptance_index]
                marker_state_visit_count = self.marker_state_visit_counts[acceptance_index]

                acceptance_score = marker_utility_arr[0] + self.marker_exploration_term*math.sqrt(math.log(marker_state_visit_count+1)/(marker_visits_arr[0]+1))
                rejection_score = marker_utility_arr[1] + self.marker_exploration_term*math.sqrt(math.log(marker_state_visit_count+1)/(marker_visits_arr[1]+1))

                is_accepted = acceptance_score >= rejection_score
                if mode == 'train':
                    self.marker_visits.append((acceptance_index, int(is_accepted)))

                if is_accepted:
                    return {'speaker_id' : self.name, 'text' : 'Accept-Deal', 'is_marker' : True,
                            'emotion' : None, 'intent' : None, 'proposal' : None}
                else:
                    return {'speaker_id' : self.name, 'text' : 'Reject-Deal', 'is_marker' : True,
                            'emotion' : None, 'intent' : None, 'proposal' : None}

            elif input_dict['text']=='Reject-Deal':
                ## Use the last conversation again to regenerate a proposal
                input_dict = self.history

        current_state = self.get_state_from_dict(input_dict)
        current_state_index = self.state_to_index(current_state)

        utility_arr = self.utility_space[current_state_index, :]
        visits_arr = self.state_action_visit_counts[current_state_index, :]
        state_visit_count = self.state_visit_counts[current_state_index, 0]

        indices = visits_arr.nonzero()
        indices = indices[1]

        all_indices = set(range(np.prod(self.state_space_dim)))
        exploration_indices = np.random.choice(list(all_indices - set(indices)), size=10)

        indices = np.concatenate((indices, exploration_indices))

        best_score = -1e10
        best_ind_arr = []
        for ind in indices:
            utility_value = utility_arr[0, ind]
            visits_count = visits_arr[0, ind]
            if mode == 'train':
                score = utility_value + self.exploration_term*math.sqrt(math.log(state_visit_count+1)/(visits_count+1))
            else:
                score = utility_value
            if score == best_score:
                best_ind_arr.append(ind)
            if score > best_score:
                best_score = score
                best_ind_arr = [ind]

        best_action = np.random.choice(best_ind_arr)

        if mode == 'train':
            self.trial_visits.append((current_state_index, best_action))

        output_dict = self.get_dict_from_state(self.index_to_state(best_action))

        self.history = input_dict
        if input_dict['proposal'] == output_dict['proposal'] and not incomplete_proposal(input_dict['proposal']):
            ## It seems like something has been agreed upon. Submit a deal
            return {'speaker_id' : self.name, 'text' : 'Submit-Deal', 'is_marker' : True,
                    'emotion' : None, 'intent' : None, 'proposal' : output_dict['proposal']}

        return output_dict

    def step_reward(self, reward):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        for ele in self.trial_visits:
            self.state_visit_counts[ele[0], 0] += 1
            self.state_action_visit_counts[ele[0], ele[1]] += 1
            self.utility_space[ele[0], ele[1]] += reward

        for ele in self.marker_visits:
            self.marker_state_visit_counts[ele[0], 0] += 1
            self.marker_action_visit_counts[ele[0], ele[1]] += 1
            self.marker_utility_space[ele[0], ele[1]] += reward

    def set_priority(self, priorities):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        sort_by = ["High", "Medium", "Low"]
        priorities = {k: priorities[k] for k in sort_by}
        self.priorities = priorities

    def get_state_from_dict(self, inpdict):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        state = []
        state.append(inpdict['emotion'])
        state.extend(inpdict['intent'])
        state.extend(convert_proposal_to_arr(inpdict['proposal'], self.priorities))

        return state

    def get_dict_from_state(self, state):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        outdict = {}
        outdict['speaker_id'] = self.name
        outdict['text'] = 'MCTS Agent does not generate text.'
        outdict['is_marker'] = False
        outdict['emotion'] = state[0]
        outdict['intent'] = state[1:11]
        proposal = {}
        proposal[self.priorities["High"]] = state[-3]
        proposal[self.priorities["Medium"]] = state[-2]
        proposal[self.priorities["Low"]] = state[-1]
        outdict['proposal'] = proposal

        return outdict

    def state_to_index(self, state):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        index = 0
        mult = np.prod(self.state_space_dim)
        for ele, ref in zip(state, self.state_space_dim):
            mult = mult//ref
            index += ele*mult

        return index

    def index_to_state(self, index):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        state = []
        mult = np.prod(self.state_space_dim)
        for ref in self.state_space_dim:
            mult = mult//ref
            state.append(index//mult)
            index = index%mult

        return state

    def save_model(self, outfile='models/mcts_v1.pkl'):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        outdict = {'state_space_dim'                : self.state_space_dim,
                   'state_visit_counts'             : self.state_visit_counts,
                   'state_action_visit_counts'      : self.state_action_visit_counts,
                   'utility_space'                  : self.utility_space,
                   'marker_state_visit_counts'      : self.marker_state_visit_counts,
                   'marker_action_visit_counts'     : self.marker_action_visit_counts,
                   'marker_utility_space'           : self.marker_utility_space,
                   'exploration_term'               : self.exploration_term,
                   'marker_exploration_term'        : self.marker_exploration_term}

        with open(outfile, 'wb') as fp:
            pickle.dump(outdict, fp)

    def load_model(self, infile='models/mcts_v1.pkl'):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        with open(infile, 'rb') as fp:
            indict = pickle.load(fp)

        self.state_space_dim                = indict['state_space_dim']
        self.state_visit_counts             = indict['state_visit_counts']
        self.state_action_visit_counts      = indict['state_action_visit_counts']
        self.utility_space                  = indict['utility_space']
        self.marker_state_visit_counts      = indict['marker_state_visit_counts']
        self.marker_action_visit_counts     = indict['marker_action_visit_counts']
        self.marker_utility_space           = indict['marker_utility_space']
        self.exploration_term               = indict['exploration_term']
        self.marker_exploration_term        = indict['marker_exploration_term']

    def start_conversation(self):
        #### NEED UPDATES!! CURRENTLY ONLY COPIED
        self.history = None
        self.priorities = None
        self.name = None
        self.conversation = None
        self.trial_visits = []
        self.marker_visits = []
        return
