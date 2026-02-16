''' NVC rule (sub-)processes.
    Decides if a syllogism yields NVC direcly
    Realizes a function: syllog --> NVC / No NVC

    In the paper, those are part of the second phase (Response Candidate Generation),
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
import pandas as pd
from typing import List
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

class NVCDirectRule(base.CognitiveSubprocess):
    def __init__(self, name):
        super(NVCDirectRule, self).__init__(name)
    
    def get_name(self):
        return "NVCDirectRule.{}".format(self.name)

    def apply(self, state_info):
        if self.is_nvc(state_info["task"]):
            return {
                "phase": base.Phase.CANDIDATE_ALTERATION,
                "task": state_info["task"],
                "state": ["NVC"]
            }
        else:
            return {
                "phase": base.Phase.TASK_INTERPRETATION,
                "task": state_info["task"],
                "state": state_info["task"]
            }

    def is_nvc(self, syllog):
        raise NotImplementedError()

    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.TASK_INTERPRETATION

class TwoSome(NVCDirectRule):
    ''' Realizes the "Two Some Rule", which allows to derive NVC from two
    non-universal quantifiers.

    The rule is part of the model TransSet.

    For additional information, see
    - Brand, et al., (2020). Extending TransSet: An Individualized Model for Human Syllogistic Reasoning.
    - Riesterer et al., (2020). Modeling Human Syllogistic Reasoning: The Role of “No Valid Conclusion”
    '''
    def __init__(self):
        super(TwoSome, self).__init__("TwoSome")

    def is_nvc(self, syllog):
        tests = ["I", "O"]
        q1 = syllog[0]
        q2 = syllog[1]
        return (q1 in tests) and (q2 in tests)

class TwoNegative(NVCDirectRule):
    ''' Realizes the "Two Negative Rule", which allows to derive NVC from two
    negative quantifiers.

    The rule is part of the model TransSet.

    For additional information, see
    - Brand, et al., (2020). Extending TransSet: An Individualized Model for Human Syllogistic Reasoning.
    - Riesterer et al., (2020). Modeling Human Syllogistic Reasoning: The Role of “No Valid Conclusion”
    '''
    def __init__(self):
        super(TwoNegative, self).__init__("TwoNegative")

    def is_nvc(self, syllog):
        tests = ["E", "O"]
        q1 = syllog[0]
        q2 = syllog[1]
        return (q1 in tests) and (q2 in tests)

class TransSetFirstNeg(NVCDirectRule):
    ''' Realizes a rule that derives NVC when the first premise quantifier is negative,
    and the other premise's quantifier is not All

    The rule is part of the model TransSet.

    For additional information, see
    - Brand, et al., (2020). Extending TransSet: An Individualized Model for Human Syllogistic Reasoning.
    '''
    def __init__(self):
        super(TransSetFirstNeg, self).__init__("FirstNeg")

    def is_nvc(self, syllog):
        q1 = syllog[0]
        q2 = syllog[1]
        return (q1 in ["E", "O"]) and (q2 != "A")

class TransSetPartNeg(NVCDirectRule):
    ''' Combines Two Some and Two Neg

    For additional information, see
    - Riesterer et al., (2020). Modeling Human Syllogistic Reasoning: The Role of “No Valid Conclusion”
    '''
    def __init__(self):
        super(TransSetPartNeg, self).__init__("PartNeg")

    def is_nvc(self, syllog):
        q1 = syllog[0]
        q2 = syllog[1]
        return (q1 != "A") and (q2 != "A")

class TransSetPath(NVCDirectRule):
    ''' Derives NVC when a transitive path could not be established based on TransSet's conversion.

    For additional information, see
    - Brand, et al., (2020). Extending TransSet: An Individualized Model for Human Syllogistic Reasoning.
    '''
    def __init__(self):
        super(TransSetPath, self).__init__("Path")
        self.ordering = { 'A' : 3, 'E' : 2, 'I' : 1, 'O' : 1}

    def is_nvc(self, syllog):
        figure = syllog[2]
        q1 = syllog[0]
        q2 = syllog[1]
        if figure == "1":
            return False
        elif figure == "2":
            return False
        else:
            return self.ordering[q1] == self.ordering[q2]

def getProcesses() -> List[NVCDirectRule]:
    return [TwoSome(), TwoNegative(), TransSetFirstNeg(), TransSetPartNeg(), TransSetPath()]