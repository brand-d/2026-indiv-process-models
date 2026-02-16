''' Preferences (sub-)processes.
    Removes conclusion candidates - as long as some remain
    Realizes a function: syllog, set(conclusions) --> set(conclusions)

    In the paper, those are part of the third phase (response candidate alteration), 
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
from typing import List

class Preference(base.CognitiveSubprocess):
    def __init__(self, name):
        super(Preference, self).__init__(name)
    
    def get_name(self):
        return "Preference.{}".format(self.name)

    def apply(self, state_info):
        selected = self.select(state_info["task"], state_info["state"])
        return {
            "phase": base.Phase.CANDIDATE_ALTERATION,
            "task": state_info["task"],
            "state": selected
        }

    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.CANDIDATE_ALTERATION

    def select(self, syllog, conclusions):
        raise NotImplementedError()

class Grice(Preference):
    ''' Realizes a preference for universal quantifers.
    It represents the Gricean maxime of quantity
    For additional information and motivation, see:
        "Grice, (1975). Logic and Conversation, 41-58."
    '''
    def __init__(self):
        super(Grice, self).__init__("Grice")

    def select(self, syllog, conclusions):
        only_univ = [x for x in conclusions if x[0] in ["A", "E"]]
        if only_univ:
            return only_univ
        else:
            return conclusions
    
    def get_name(self):
        return super().get_name()

class Matching(Preference):
    ''' Implements a preference for "matching" quantifiers, i.e., conclusion quantifiers
    that also occur in the premises.

    Based on the Matching Hypothesis
    Wetherick & Gilhooly, (1995). ‘Atmosphere’, matching, and logic in syllogistic reasoning.

    Note that when deriving a quantifier, Matching also proposes a order among quantifiers that is not 
    considered here.
    '''
    def __init__(self):
        super(Matching, self).__init__("Matching")

    def select(self, syllog, conclusions):
        premise_quants = [syllog[0], syllog[1]]
        in_premise = [x for x in conclusions if x[0] in premise_quants]
        if in_premise:
            return in_premise
        else:
            return conclusions

class Atmosphere(Preference):
    ''' Implements a preference for conclusions in line with the atmosphere of the syllogism, i.e.,
    the quantifiers that the Atmosphere theory would conclude from the premises.

    For more information, see:
    Woodworth & Sells, (1935). An atmosphere effect in formal syllogistic reasoning.

    '''
    def __init__(self):
        super(Atmosphere, self).__init__("Atmosphere")

    def select(self, syllog, conclusions):
        atmosphere_quants = self.get_quantifier(syllog)
        in_premise = [x for x in conclusions if x[0] in atmosphere_quants]
        if in_premise:
            return in_premise
        else:
            return conclusions
        
    def get_quantifier(self, syllog):
        first = syllog[0]
        second = syllog[1]
        premises = ''.join(sorted([first, second]))
        responses = []
        if premises == 'AA':
            return ["A"]
        elif premises == 'AI':
            return ["I"]
        elif premises == 'AE':
            return ["E"]
        elif premises == 'AO':
            return ["O"]
        elif premises == 'EE':
            return ["E"]
        elif premises == 'EI':
            return ["O"]
        elif premises == 'EO':
            return ["O"]
        elif premises == 'II':
            return ["I"]
        elif premises == 'IO':
            return ["O"]
        elif premises == 'OO':
            return ["O"]
        return responses

class PHMInformativeness(Preference):
    ''' Implements a preference for conclusions that are most informative according to
    the Probability Heuristics Model (PHM).
    It follows and ordering of A > I > E > O

    For more information, see:
    - Chater & Oaksford, (1999): The Probability Heuristics Model of Syllogistic Reasoning
    - Oaksford & Chater, (2001): The probabilistic approach to human reasoning
    '''
    def __init__(self):
        super(PHMInformativeness, self).__init__("PHMInformativeness")

    def select(self, syllog, conclusions):
        quants = set([x[0] for x in conclusions])
        for q in ["A", "I", "E", "O"]:
            if q in quants:
                return [x for x in conclusions if x[0] == q]
        return conclusions

class FigureEffect(Preference):
    ''' Implements a preference for conclusions that are in line with the Figure effect.
     The Figure Effect is a bias, which was found to cause human reasoners to prefer
    "ac" directions for figure 1 syllogisms, and "ca" for figure 2, respectively.

    For more information, refer to
    - Dickstein, (1978). The effect of figure on syllogistic reasoning
    - Johnson-Laird & Bara, (1984). Syllogistic inference
    '''
    def __init__(self):
        super(FigureEffect, self).__init__("FigureEffect")

    def select(self, syllog, conclusions):
        ok = None
        if syllog[2] == "1":
            ok = [x for x in conclusions if x[1:] == "ac"]
        elif syllog[2] == "2":
            ok = [x for x in conclusions if x[1:] == "ca"]
        else:
            ok = conclusions
        
        if ok:
            return ok
        return conclusions

def getProcesses() -> List[Preference]:
    return [Grice(), Atmosphere(), Matching(), PHMInformativeness(), FigureEffect()]