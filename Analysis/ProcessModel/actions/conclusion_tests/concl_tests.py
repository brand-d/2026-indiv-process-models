''' Conclusion Test (sub-)processes.
    Removes conclusion candidates - and yields NVC if all are removed
    Realizes a function: syllog, set(conclusions) --> set(conclusions) | NVC
    
    In the paper, those are part of the third phase (response candidate alteration), 
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
import pandas as pd
from typing import List
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

class ConclusionTest(base.CognitiveSubprocess):
    ''' Superclass for all Conclusions Tests.
    '''
    def __init__(self, name):
        super(ConclusionTest, self).__init__(name)

    def get_name(self):
        return "ConclusionTest.{}".format(self.name)

    def apply(self, state_info):
        concls = self.test(state_info["task"], state_info["state"])
        if not concls:
            concls = ["NVC"]
        return {
            "phase": base.Phase.CANDIDATE_ALTERATION,
            "task": state_info["task"],
            "state": concls
        }

    def test(self, syllog, conclusions):
        raise NotImplementedError()
    
    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.CANDIDATE_ALTERATION

class OHeuristic(ConclusionTest):
    ''' Implements the O-Heuristic of the Probability Heuristics Model (PHM).
    For more information about PHM, please refer to 
    Chater, & Oaksford (1999). The Probability Heuristics Model of Syllogistic Reasoning

    Thereby, O is generally avoided and leads to NVC, as suggested by 
    Copeland (2006). Theories of categorical reasoning and extended syllogisms
    '''
    def __init__(self):
        super(OHeuristic, self).__init__("O-Heuristic")

    def test(self, syllog, conclusions):
        return ["NVC" if (x == "NVC") or (x[0] == "O") else x for x in conclusions]

class MMT(ConclusionTest):
    ''' Implements a conclusion test based on the Mental Model Theory (MMT).
    See 
    - Johnson-Laird (1983). Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness

    MMT is implemented based on a prediction table (taken from Khemlani, S. S.; and Johnson-Laird, P. N. 2012. Theories
    of the syllogism: A meta-analysis).
     
    The conclusion test works as follows:
    - whenever NVC is part of the conclusion table for a syllogism, it is responded with NVC
    - If a conclusion candidate is not part of the conclusions of MMT, it is omitted
    '''
    def __init__(self):
        super(MMT, self).__init__("MMT")
        pred_df = pd.read_csv(os.path.join(package_directory, "..", 'MMT.csv'))
        self.predictions = dict(zip(pred_df['Syllogism'].tolist(),
                            [x.split(';') for x in pred_df['Prediction']]))

    def test(self, syllog, conclusions):
        responses = self.predictions[syllog]
        if "NVC" in responses:
            return ["NVC"]
        return ["NVC" if x not in responses else x for x in conclusions]

class MaxHeuristics(ConclusionTest):
    ''' Implements the Max-heuristic of the Probability Heuristics Model (PHM).
    For more information about PHM, please refer to 
    - Chater, & Oaksford (1999). The Probability Heuristics Model of Syllogistic Reasoning

    Thereby, if the confidence is too low in the quantifier of the max-Premise,
    an NVC response is generated, as suggested by 
    - Copeland (2006). Theories of categorical reasoning and extended syllogisms

    In order to account for various confidence levels, three subclasses are implemented:
    MaxHeuristicA: Only confident in A
    MaxHeuristicI: Only confident in I and A
    MaxHeuristicE: Only confident in E, I and A
    '''
    def __init__(self, quantifier):
        super(MaxHeuristics, self).__init__("MaxHeuristic: {}".format(quantifier))
        self.quantifier = quantifier
        self.informativeness = ["A", "I", "E", "O"]
        
    def max_premise(self, syllog):
        ordering = self.informativeness[::-1]
        if ordering.index(syllog[0]) < ordering.index(syllog[1]):
            return syllog[1]
        else:
            return syllog[0]
    
    def test(self, syllog, conclusions):
        target_idx = self.informativeness.index(self.quantifier)
        max_prem = self.max_premise(syllog)
        return ["NVC" if (x == "NVC") or (self.informativeness.index(max_prem) > target_idx) else x for x in conclusions]

class MaxHeuristicA(MaxHeuristics):
    def __init__(self):
        super(MaxHeuristicA, self).__init__("A")
        
class MaxHeuristicI(MaxHeuristics):
    def __init__(self):
        super(MaxHeuristicI, self).__init__("I")

class MaxHeuristicE(MaxHeuristics):
    def __init__(self):
        super(MaxHeuristicE, self).__init__("E")

def get_process_combinations() -> List[List[ConclusionTest]]:
    ''' Returns the combination of subprocesses that are possible.
    This is done due to some of them dominating each other, and therefore
    are not necessary to be tested together
    '''
    return [
        [OHeuristic()],
        [OHeuristic(), MaxHeuristicA()],
        [OHeuristic(), MaxHeuristicI()],
        [OHeuristic(), MaxHeuristicE()],
        [MaxHeuristicA()],
        [MaxHeuristicI()], 
        [MaxHeuristicE()],
        [MMT(), OHeuristic()],
        [MMT(), OHeuristic(), MaxHeuristicA()],
        [MMT(), OHeuristic(), MaxHeuristicI()],
        [MMT(), OHeuristic(), MaxHeuristicE()],
        [MMT(), MaxHeuristicA()],
        [MMT(), MaxHeuristicI()], 
        [MMT(), MaxHeuristicE()],
        [MMT()]
    ]

def getProcesses() -> List[ConclusionTest]:
    return [OHeuristic(), MaxHeuristicA(), MaxHeuristicI(), MaxHeuristicE(), MMT()]
