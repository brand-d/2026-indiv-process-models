''' Direction determination (sub-)processes.
    Realizes a function: syllog --> set(conclusion_quantifiers)

    In the paper, those are part of the second phase (Response Candidate Generation),
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
from typing import List
import pandas as pd
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

class QuantifierGenerator(base.CognitiveSubprocess):
    def __init__(self, name):
        super(QuantifierGenerator, self).__init__(name)
    
    def get_name(self):
        return "QuantifierGenerator.{}".format(self.name)

    def apply(self, state_info):
        if not self.is_applicable(state_info["state"]):
            return state_info
        return {
            "phase": base.Phase.CANDIDATE_GENERATION,
            "task": state_info["task"],
            "state": self.get_quantifier(state_info["state"])
        }

    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.TASK_INTERPRETATION

    def get_quantifier(self, syllog):
        raise NotImplementedError()

    def is_applicable(self, syllog):
        raise NotImplementedError()

class MMT(QuantifierGenerator):
    ''' Implements a quantifier selection behavior based on the Mental Model Theory (MMT).
    See 
    - Johnson-Laird, (1983). Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness

    MMT is implemented based on a prediction table 
    (taken from Khemlani & Johnson-Laird, (2012). Theories of the syllogism: A meta-analysis).
    
    The quantifier is suggested by the prediction table is used as the result of this action.
    '''
    def __init__(self):
        super(MMT, self).__init__("MMT")
        pred_df = pd.read_csv(os.path.join(package_directory, "..", 'MMT.csv'))
        self.predictions = dict(zip(pred_df['Syllogism'].tolist(),
                            [x.split(';') for x in pred_df['Prediction']]))

    def get_quantifier(self, syllog):
        responses = self.predictions[syllog]
        quantifiers = set([x[0] for x in responses if x != "NVC"])
        return list(quantifiers)

    def is_applicable(self, syllog):
        return True

class Atmosphere(QuantifierGenerator):
    ''' Returns partial conclusions in line with the atmosphere of the syllogism, i.e.,
    the quantifiers that the Atmosphere theory would conclude from the premises.

    Essentially, those combine the negativity and particularity of the two premises
    (i.e., using negativity and particularity, if present in either premise).
    
    For more information, see:
    Woodworth & Sells, (1935). An atmosphere effect in formal syllogistic reasoning.

    It is also the mechanism TransSet uses to infer the quantifer, see
    - Brand, et al. (2020): Extending TransSet: An Individualized Model for Human Syllogistic Reasoning
    '''
    def __init__(self):
        super(Atmosphere, self).__init__("Atmosphere")
 
    def is_applicable(self, syllog):
        return True

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

class Matching(QuantifierGenerator):
    ''' Selects a partial conclusion quantifier based on "matching" quantifiers, i.e., conclusion quantifiers
    that also occur in the premises.

    Based on the Matching Hypothesis
    Wetherick & Gilhooly, (1995). ‘Atmosphere’, matching, and logic in syllogistic reasoning.

    Thereby, an ordering of E > I = O > A is used, to minimize the "amount of entities" the conclusion makes
    a statement about (considering E as "no elements", and not "All elements not").
    '''
    def __init__(self):
        super(Matching, self).__init__("Matching")

    def get_quantifier(self, syllog):
        quants = [syllog[0], syllog[1]]
        if "E" in quants: 
            return ["E"]
        elif "I" in quants:
            return ["I", "O"]
        elif "O" in quants:
            return ["I", "O"]
        else: 
            return ["A"]

    def is_applicable(self, syllog):
        return True

class PHM(QuantifierGenerator):
    ''' Selects quantifiers in line with the Probability Heuristics Model (PHM).
    Thereby, it uses the min-heuristic: choose the quantifier of the conclusion to match the quantifier 
    in the least informative premise (the min-premise). The informativeness thereby is: A > I > E > O

    For more information about PHM, please refer to 
    - Chater & Oaksford, (1999): The Probability Heuristics Model of Syllogistic Reasoning
    - Oaksford & Chater, (2001): The probabilistic approach to human reasoning

    Implementation is adapted from the implementation used in 
    Riesterer et al., (2020): Do Models Capture Individuals? Evaluating Parameterized Models for Syllogistic Reasoning
    '''
    def __init__(self):
        super(PHM, self).__init__("PHM")
        
        self.informativeness = ["A", "I", "E", "O"]

    def get_quantifier(self, syllog):
        prems = self.getpremises(syllog)
        prem1 = prems["prem1"]
        prem2 = prems["prem2"]
        minPrem = prems["prem1"]
        maxPrem = prems["prem2"]

        if self.informativeness.index(minPrem[0]) < self.informativeness.index(maxPrem[0]):
            minPrem = prem2
            maxPrem = prem1
        
        minConclCands = [[minPrem[0], "A", "C"], [minPrem[0], "C", "A"]]
        minConclCandPhrases = [self.noun_phrase(x) for x in minConclCands]
        
        prem1Phrase = self.noun_phrase(prem1)
        prem2Phrase = self.noun_phrase(prem2)
        premPhrases = [prem1Phrase, prem2Phrase]
        
        return [minPrem[0]]
    
    def getpremises(self, syllog) -> dict:
        figure = syllog[2]
        prem1 = None
        prem2 = None
        if figure == "1":
            prem1 = [syllog[0], 'A', 'B']
            prem2 = [syllog[1], 'B', 'C']
        elif figure == "2":
            prem1 = [syllog[0], 'B', 'A']
            prem2 = [syllog[1], 'C', 'B']
        elif figure == "3":
            prem1 = [syllog[0], 'A', 'B']
            prem2 = [syllog[1], 'C', 'B']
        elif figure == "4":
            prem1 = [syllog[0], 'B', 'A']
            prem2 = [syllog[1], 'B', 'C']
        return {
            "prem1": prem1,
            "prem2": prem2
        }

    def noun_phrase(self, premise):
        return [premise[0].replace('O', 'I'), premise[1]]

    def is_applicable(self, syllog):
        return True

def getProcesses() -> List[QuantifierGenerator]:
    return [Atmosphere(), Matching(), MMT(), PHM()]