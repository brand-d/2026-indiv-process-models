''' Extension (sub-)processes.
    Introduces additional conclusion candidates.
    Realizes a function: syllog, set(conclusions) --> set(conclusions)

    In the paper, those are part of the third phase (response candidate alteration), 
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
from typing import List

class Extension(base.CognitiveSubprocess):
    def __init__(self, name):
        super(Extension, self).__init__(name)
    
    def get_name(self):
        return "Extension.{}".format(self.name)
    
    def apply(self, state_info):
        return {
            "phase": base.Phase.CANDIDATE_ALTERATION,
            "task": state_info["task"],
            "state": self.extend(state_info["task"], state_info["state"])
        }

    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.CANDIDATE_ALTERATION

    def extend(self, syllog, conclusions):
        raise NotImplementedError()

            
class PEntailment(Extension):
    ''' Implements the p-entailment of the Probability Heuristics Model (PHM).
    For more information about PHM, please refer to 
    - Chater & Oaksford, (1999): The Probability Heuristics Model of Syllogistic Reasoning
    - Oaksford & Chater, (2001): The probabilistic approach to human reasoning

    It adds conclusions that are probabilistically entailed, e.g., "I" is entailed in "A"
    '''
    def __init__(self):
        super(PEntailment, self).__init__("PEntailment")

    def extend(self, syllog, conclusions):
        new_concl = set(conclusions)
        for concl in conclusions:
            if concl[0] == "A":
                new_concl.add("I{}".format(concl[1:]))
            elif concl[0] == "E":
                new_concl.add("O{}".format(concl[1:]))
            elif concl[0] == "I":
                new_concl.add("O{}".format(concl[1:]))
            elif concl[0] == "O":
                new_concl.add("I{}".format(concl[1:]))
        return new_concl

class Grice(Extension):
    ''' Implements an pragmatic gricean implicature:
    "Some A are B" also means "Some A are not B" and vice versa.
    
    While not logically valid, it is a pragmatic interpetation which humans
    rely on. i.e., a majority stated that "Some" can not mean "All" in 
    Brand, & Ragni, (2023). Effect of Response Format on Syllogistic Reasoning
    '''
    def __init__(self):
        super(Grice, self).__init__("Grice")

    def extend(self, syllog, conclusions):
        new_concl = set(conclusions)
        for concl in conclusions:
            if concl[0] == "I":
                new_concl.add("O{}".format(concl[1:]))
            elif concl[0] == "O":
                new_concl.add("I{}".format(concl[1:]))
        return new_concl

def getProcesses() -> List[Extension]:
    return [PEntailment(), Grice()]