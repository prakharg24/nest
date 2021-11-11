import copy
import math
import numpy as np
import json
import pickle
from agent_utils import get_random_emotion, get_random_intent
from agent_utils import get_proposal_score, incomplete_proposal, switch_proposal_perspective, normalize_prob, choose_random_with_prob
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

    def step_active(self, input_dict, mode='eval'):
        ### Skeleton Function. Inherit this and change to return some sensible proposal
        return input_dict

    def step_passive(self, input_dict, output_dict, mode='train'):
        ### Skeleton Function. Inherit this and change to memorise the agent's history
        return

    def start_conversation(self):
        ### Use this function to reset any parameters if required before starting a brand new conversation
        return

### Required for training of agents which use actual dataset based calculations
class AgentDummy(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)

    def step_active(self, input_dict, mode=None):
        ## Switch perspective since we are looking for other speaker's utterance
        input_dict = switch_proposal_perspective(input_dict)
        for ind, ele in enumerate(self.conversation):
            if ele == input_dict:
                curr_dia = ind
                break

        return self.conversation[curr_dia + 1]

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
                high_index = input_dict['proposal'][self.priorities["High"]]
                medium_index = input_dict['proposal'][self.priorities["Medium"]]
                low_index = input_dict['proposal'][self.priorities["Low"]]
                self.acceptance_count[high_index, medium_index, low_index] += 1
            return

        self.set_emotion_counts(input_dict, output_dict)
        self.set_intent_counts(input_dict, output_dict)
        self.set_proposal_counts(input_dict, output_dict)

    def step_active(self, input_dict, mode='eval'):
        if mode == 'train':
            ## Switch perspective since we are looking for other speaker's utterance
            input_dict = switch_proposal_perspective(input_dict)
            for ind, ele in enumerate(self.conversation):
                if ele == input_dict:
                    curr_dia = ind
                    break
            self.step_passive(input_dict, self.conversation[ind+1], mode=mode)
            return self.conversation[ind+1]

        if input_dict is None:
            ## The case when the agent needs to speak first
            raise Exception("Input Dict was Empty. The Agents are not trained to start negotiations")

        if input_dict['is_marker']:
            ## Marker cases
            if input_dict['text']=='Submit-Deal':
                high_index = input_dict['proposal'][self.priorities["High"]]
                medium_index = input_dict['proposal'][self.priorities["Medium"]]
                low_index = input_dict['proposal'][self.priorities["Low"]]
                acceptance_prob = sum(self.acceptance_count[:high_index, :medium_index, :low_index])/sum(self.acceptance_count)
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
        input_proposal_arr = [input_dict['proposal'][self.priorities["High"]],
                              input_dict['proposal'][self.priorities["Medium"]],
                              input_dict['proposal'][self.priorities["Low"]]]
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

        if input_dict['proposal'] == out_proposal_dict and not incomplete_proposal(out_proposal_dict):
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
        input_arr = [input_dict['proposal'][self.priorities[ele]] for ele in self.priorities]
        output_arr = [output_dict['proposal'][self.priorities[ele]] for ele in self.priorities]

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


class AgentMCTS(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)
        self.utility_space = np.zeros((num_emotion, num_intent, 2, 4, 4, 4))
        self.visit_counts = np.ones((num_emotion, num_intent, 2, 4, 4, 4))
        self.trial_visits = []
        self.history = None

    def step_active(self, input_dict, mode='eval'):
        ### Skeleton Function. Inherit this and change to return some sensible proposal
        return input_dict

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
                high_index = input_dict['proposal'][self.priorities["High"]]
                medium_index = input_dict['proposal'][self.priorities["Medium"]]
                low_index = input_dict['proposal'][self.priorities["Low"]]
                self.acceptance_count[high_index, medium_index, low_index] += 1
            return

        self.set_emotion_counts(input_dict, output_dict)
        self.set_intent_counts(input_dict, output_dict)
        self.set_proposal_counts(input_dict, output_dict)
        return

    def set_priority(self, priorities):
        sort_by = ["High", "Medium", "Low"]
        priorities = {k: priorities[k] for k in sort_by}
        self.priorities = priorities
        self.fill_heuristic_utility()

    def fill_heuristic_utility(self):
        it = np.nditer(self.utility_space, flags=['multi_index'], op_flags=['readwrite'])
        for ele in it:
            curr_ind = it.multi_index
            proposal = {}
            proposal[self.priorities["High"]] = curr_ind[3]
            proposal[self.priorities["Medium"]] = curr_ind[4]
            proposal[self.priorities["Low"]] = curr_ind[5]

            score = get_proposal_score(self.priorities, proposal, self.score_weightage)
            ele[...] = score

    def save_model(self, outfile='some_fixed_file.pt'):
        ## Save the model parameters/dict etc. so that it can be easily laoded
        return

    def load_model(self, infile='some_fixed_file.pt'):
        ## load the model parameters back
        return

    def start_conversation(self):
        self.history = None
        self.priorities = None
        self.name = None
        self.conversation = None
        self.trial_visits = []
        return
