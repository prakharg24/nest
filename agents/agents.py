import copy
import math
from task_utils import get_random_emotion, get_random_intent
from task_utils import get_proposal_score, incomplete_proposal

### define agents
class AgentTabular():
    def __init__(self, score_weightage, length_penalty, id):
        ## Add all initializations as required
        self.score_weightage = score_weightage
        self.length_penalty = length_penalty
        self.id = id

    def set_priority(self, priorities):
        self.priorities = priorities

    def step(self, input_emotion, input_intent, input_proposal):
        ### Skeleton Function. Inherit this and change to return some sensible proposal
        return input_proposal

    def step_passive(self, input_emotion, input_intent, input_proposal, user_id):
        ### Skeleton Function. Inherit this and change to memorise the agent's history
        return

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
