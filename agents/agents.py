import copy
import math
import numpy as np
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
        sort_by = ["High", "Medium", "Low"]
        priorities = {k: priorities[k] for k in sort_by}
        self.priorities = priorities

    def set_name(self, agent_name):
        self.name = agent_name

    def set_conversation(self, conversation):
        self.conversation = conversation

    def step(self, input_dict, mode='train'):
        ### Skeleton Function. Inherit this and change to return some sensible proposal
        return input_dict

    def step_passive(self, input_dict, output_dict, mode='train'):
        ### Skeleton Function. Inherit this and change to memorise the agent's history
        return

### Required for training of agents which use actual dataset based calculations
class AgentDummy(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)

    def step_passive(self, input_dict, output_dict, mode=None):
        ### Do nothing since its the dummy agent
        return

    def step(self, input_dict, mode=None):
        for ind, ele in enumerate(self.conversation):
            if ele == input_dict:
                curr_dia = ind
                break

        return self.conversation[curr_dia + 1]

class AgentNoPlanningBayesian(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)
        self.seen = 0 ## normalizer for probabilities
        self.emotion_count = np.array([0 for _ in range(num_emotion)]) ## probability of each emotion
        self.emotion_trans_count = np.zeros((num_emotion, num_emotion)) ## transition probability between emotions
        self.intent_count = np.array([0 for _ in range(num_intent)]) ## probability of each intent
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

        self.proposal_float = {"Firewood": -1, "Water": -1, "Food": -1}

    def prepare_proposal(self):
        proposal_out = {}
        for ele in self.proposal_float:
            if (ele != -1):
                proposal_out[ele] = math.ceil(self.proposal_float[ele])
            else:
                proposal_out[ele] = -1
        return proposal_out

    def step_passive(self, input_dict, output_dict, mode='train'):
        if mode == 'eval':
            ## If the agent is not in training mode, no need to record observations
            return
        if input_dict is None:
            ## Just the first spoken dialogue. Skip
            return
        if input_dict['emotion'] is None or output_dict['emotion'] is None:
            ## Marker statements. Skip
            return
        if incomplete_proposal(input_dict['proposal']) or incomplete_proposal(output_dict['proposal']):
            ## Some part of proposal unknown. Skip
            return

        input_dict['proposal'] = switch_proposal_perspective(input_dict['proposal'])
        self.seen += 1
        print("Bayesian Agent Seen Examples : ", self.seen)

        self.set_emotion_counts(input_dict, output_dict)
        self.set_intent_counts(input_dict, output_dict)
        self.set_proposal_counts(input_dict, output_dict)

    def step(self, input_dict, mode='train'):
        if mode=='train':
            for ind, ele in enumerate(self.conversation):
                if ele == input_dict:
                    curr_dia = ind
                    break
            self.step_passive(input_dict, self.conversation[ind+1], mode=mode)
            return self.conversation[ind+1]

        else:
            emotion_choice_arr = normalize_prob(self.emotion_trans_count[:, input_dict['emotion']])
            out_emotion = choose_random_with_prob(range(num_emotion), emotion_choice_arr)

            out_intent = []
            for ite in range(num_intent):
                intent_choice_arr = normalize_prob(self.intent_trans_count[ite, :, input_dict['intent'][ite]])
                out_intent.append(choose_random_with_prob(range(2), intent_choice_arr))

            out_proposal = []
            input_proposal_arr = [input_dict['proposal'][self.priorities["High"]],
                                  input_dict['proposal'][self.priorities["Medium"]],
                                  input_dict['proposal'][self.priorities["Low"]]]
            for ind, priority in enumerate(self.priorities):
                if input_proposal_arr[ind]==-1:
                    out_proposal.append(np.random.choice([-1, 3-ind]))
                    continue

                prob_emotion_given_proposal = normalize_prob(self.proposal_emotion_joint_count[priority][:, out_emotion])
                prob_numerator = prob_emotion_given_proposal
                for ite in range(num_intent):
                    prob_intent_given_proposal = normalize_prob(self.proposal_intent_joint_count[priority][:, ite, out_intent[ite]])
                    prob_numerator = prob_numerator * prob_intent_given_proposal
                prob_prevproposal_given_proposal = normalize_prob(self.proposal_prevproposal_joint_count[priority][:, input_proposal_arr[ind]])
                prob_numerator = prob_numerator * prob_prevproposal_given_proposal

                proposal_choice_arr = normalize_prob(prob_numerator)
                out_proposal.append(choose_random_with_prob(range(4), proposal_choice_arr))

            

    def set_emotion_counts(self, input_dict, output_dict):
        self.emotion_count[output_dict['emotion']] += 1
        self.emotion_trans_count[input_dict['emotion'], output_dict['emotion']] += 1

    def set_intent_counts(self, input_dict, output_dict):
        self.intent_count = np.array([e1 + e2 for e1, e2 in zip(self.intent_count, output_dict['intent'])])

        for counter, (e1, e2) in enumerate(zip(input_dict['intent'], output_dict['intent'])):
            self.intent_trans_count[counter, e1, e2] += 1

    def set_proposal_counts(self, input_dict, output_dict):
        input_arr = [input_dict['proposal'][self.priorities["High"]],
                     input_dict['proposal'][self.priorities["Medium"]],
                     input_dict['proposal'][self.priorities["Low"]]]
        output_arr = [output_dict['proposal'][self.priorities["High"]],
                      output_dict['proposal'][self.priorities["Medium"]],
                      output_dict['proposal'][self.priorities["Low"]]]

        self.proposal_prevproposal_joint_count["High"][output_arr[0], input_arr[0]] += 1
        self.proposal_prevproposal_joint_count["Medium"][output_arr[1], input_arr[1]] += 1
        self.proposal_prevproposal_joint_count["Low"][output_arr[2], input_arr[2]] += 1

        self.proposal_emotion_joint_count["High"][output_arr[0], output_dict['emotion']] += 1
        self.proposal_emotion_joint_count["Medium"][output_arr[1], output_dict['emotion']] += 1
        self.proposal_emotion_joint_count["Low"][output_arr[2], output_dict['emotion']] += 1

        for counter, e1 in enumerate(output_dict['intent']):
            self.proposal_intent_joint_count["High"][output_arr[0], counter, e1] += 1
            self.proposal_intent_joint_count["Medium"][output_arr[1], counter, e1] += 1
            self.proposal_intent_joint_count["Low"][output_arr[2], counter, e1] += 1

class RandomAgentConsiderate(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)

    def set_priority_and_proposal(self, priorities):
        self.set_priority(priorities)
        self.proposal = self.set_initial_proposal()
        self.proposal_float = copy.deepcopy(self.proposal)

    def set_initial_proposal(self):
        proposal = {}
        for ele in self.priorities:
            if self.priorities[ele]=="High":
                proposal[ele] = 3
            elif self.priorities[ele]=="Medium":
                proposal[ele] = 2
            elif self.priorities[ele]=="Low":
                proposal[ele] = 1
        return proposal

    def check_acceptable(self, proposal):
        if incomplete_proposal(proposal):
            return False
        score_if_accept = get_proposal_score(self.priorities, proposal, self.score_weightage)
        score_current = get_proposal_score(self.priorities, self.proposal, self.score_weightage)

        if (score_if_accept >= score_current):
            return True
        else:
            return False

    def adjust_proposal(self, proposal):
        for ele in self.proposal_float:
            if (self.proposal_float[ele] > proposal[ele] and proposal[ele]!=-1):
                self.proposal_float[ele] = self.proposal_float[ele] - 0.2

        for ele in self.proposal_float:
            self.proposal[ele] = math.ceil(self.proposal_float[ele])

        return copy.deepcopy(self.proposal)

    def step(self, input_emotion, input_intent, input_proposal):
        is_acceptable = self.check_acceptable(input_proposal)

        out_emotion = get_random_emotion()
        out_intent = get_random_intent()

        if is_acceptable:
            out_proposal = input_proposal
        else:
            out_proposal = self.adjust_proposal(input_proposal)

        return out_emotion, out_intent, out_proposal, is_acceptable

class RandomAgentStubborn(AgentTabular):
    def __init__(self, score_weightage, length_penalty, id):
        super().__init__(score_weightage, length_penalty, id)

    def set_priority_and_proposal(self, priorities):
        self.set_priority(priorities)
        self.proposal = self.set_initial_proposal()

    def set_initial_proposal(self):
        proposal = {}
        for ele in self.priorities:
            if self.priorities[ele]=="High":
                proposal[ele] = 3
            elif self.priorities[ele]=="Medium":
                proposal[ele] = 2
            elif self.priorities[ele]=="Low":
                proposal[ele] = 1
        return proposal

    def check_acceptable(self, proposal):
        if incomplete_proposal(proposal):
            return False
        score_if_accept = get_proposal_score(self.priorities, proposal, self.score_weightage)
        score_current = get_proposal_score(self.priorities, self.proposal, self.score_weightage)

        if (score_if_accept >= score_current):
            return True
        else:
            return False

    def step(self, input_emotion, input_intent, input_proposal):
        is_acceptable = self.check_acceptable(input_proposal)

        out_emotion = get_random_emotion()
        out_intent = get_random_intent()

        if is_acceptable:
            out_proposal = input_proposal
        else:
            out_proposal = copy.deepcopy(self.proposal)

        return out_emotion, out_intent, out_proposal, is_acceptable
