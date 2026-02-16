''' Direction determination (sub-)processes.
    Realizes a function: syllog, set(conclusion_quantifiers) --> set(conclusions)

    In the paper, those are part of the second phase (Response Candidate Generation),
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
import pandas as pd
from typing import List, Collection
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

def cartesion_conclusions(quants: Collection[str], dirs: Collection[str]) -> Collection[str]:
    ''' Helper function that creates all combinations of quantifiers and directions.

    Parameters
    ----------
    quants: Collection[str]
        Collection of quantifiers
    
    dirs: Collection[str]
        Collection of directions
    
    Returns
    -------
    Collection[str]
        A colletion of conclusion candidates (quantifiers + directions)

    '''
    results = set()
    for quant in quants:
        for dir in dirs:
            results.add("{}{}".format(quant, dir))
    return results

class DirectionGenerator(base.CognitiveSubprocess):
    def __init__(self, name):
        super(DirectionGenerator, self).__init__(name)
    
    def get_name(self):
        return "DirectionGenerator.{}".format(self.name)

    def apply(self, state_info):
        if not self.is_applicable(state_info["task"], state_info["state"]):
            return state_info
        concls = self.get_directions(state_info["task"], state_info["state"])
        return {
            "phase": base.Phase.CANDIDATE_ALTERATION,
            "task": state_info["task"],
            "state": concls
        }

    def get_directions(self, syllog, quantifiers):
        raise NotImplementedError()

    def is_applicable(self, syllog, quantifiers):
        raise NotImplementedError()

    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.CANDIDATE_GENERATION
        
class All(DirectionGenerator):
    ''' Class representing bidirectional direction choices.
    '''
    def __init__(self):
        super(All, self).__init__("All")
    
    def get_directions(self, syllog, quantifiers):
        return cartesion_conclusions(quantifiers, ["ac", "ca"])

    def is_applicable(self, syllog, quantifiers):
        return True

class MMT(DirectionGenerator):
    ''' Implements a direction selection behavior based on the Mental Model Theory (MMT).
    See 
    - Johnson-Laird, (1983). Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness

    MMT is implemented based on a prediction table 
    (taken from Khemlani & Johnson-Laird, (2012). Theories of the syllogism: A meta-analysis).
    
    The direction is suggested by the prediction table is used as the result of this action.
    '''
    def __init__(self):
        super(MMT, self).__init__("MMT")
        pred_df = pd.read_csv(os.path.join(package_directory, "..", 'MMT.csv'))
        self.predictions = dict(zip(pred_df['Syllogism'].tolist(),
                            [x.split(';') for x in pred_df['Prediction']]))

    def get_directions(self, syllog, quantifiers):
        responses = self.predictions[syllog]
        pred_quants = set([x[0] for x in responses if x != "NVC"])
        result = set([x for x in responses if x[0] in quantifiers])
        
        missing_quants = [x for x in quantifiers if x not in pred_quants]
        other_concls = cartesion_conclusions(missing_quants, ["ac", "ca"])
        
        result.update(other_concls)

        return result

    def is_applicable(self, syllog, quantifiers):
        pred_quants = set([x[0] for x in self.predictions[syllog] if x != "NVC"])
        quants = set(quantifiers)
        return len(pred_quants.intersection(quants)) > 0

class FigureEffect(DirectionGenerator):
    ''' Figure effect is a bias, which was found to cause human reasoners to prefer
    "ac" directions for figure 1 syllogisms, and "ca" for figure 2, respectively.

    For more information, refer to
    - Dickstein, (1978). The effect of figure on syllogistic reasoning
    - Johnson-Laird & Bara, (1984). Syllogistic inference

    Furthermore, it the way TransSet determines its direction, after using the Conversion
    - Brand, et al. (2020): Extending TransSet: An Individualized Model for Human Syllogistic Reasoning
    '''
    def __init__(self):
        super(FigureEffect, self).__init__("FigureEffect")
    
    def is_applicable(self, syllog, quantifiers):
        return (syllog[2] == "1") or (syllog[2] == "2")
    
    def get_directions(self, syllog, quantifiers):
        if syllog[2] == "1":
            return cartesion_conclusions(quantifiers, ["ac"])
        elif syllog[2] == "2":
            return cartesion_conclusions(quantifiers, ["ca"])
        else:
            raise ValueError("Figure Effect shouldn't be applied to figure {}".format(syllog[2]))

class PHM(DirectionGenerator):
    ''' Implements the attachment-heuristic of the Probability Heuristics Model (PHM).
    For more information about PHM, please refer to 
    - Chater & Oaksford, (1999): The Probability Heuristics Model of Syllogistic Reasoning
    - Oaksford & Chater, (2001): The probabilistic approach to human reasoning

    Implementation is adapted from the implementation used in 
    Riesterer et al., (2020): Do Models Capture Individuals? Evaluating Parameterized Models for Syllogistic Reasoning
    '''
    def __init__(self):
        super(PHM, self).__init__("PHM")
        
        self.informativeness = ["A", "I", "E", "O"]

    def get_directions(self, syllog, quantifiers):
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
        
        subject = None
        if prem1[0] == prem2[0]:
            return cartesion_conclusions(quantifiers, ["ac", "ca"])
        elif (minConclCandPhrases[0] == prem1Phrase) and (minConclCandPhrases[1] not in premPhrases):
            subject = minConclCandPhrases[0][1]
        elif (minConclCandPhrases[0] == prem2Phrase) and (minConclCandPhrases[1] not in premPhrases):
            subject = minConclCandPhrases[0][1]
        elif (minConclCandPhrases[1] == prem1Phrase) and (minConclCandPhrases[0] not in premPhrases):
            subject = minConclCandPhrases[1][1]
        elif (minConclCandPhrases[1] == prem2Phrase) and (minConclCandPhrases[0] not in premPhrases):
            subject = minConclCandPhrases[1][1]
        else:
            subject = "{}{}".format(maxPrem[1], maxPrem[2]).replace("B", "")
        
        directions = [subject.replace("A", "ac").replace("C", "ca")]
        result = cartesion_conclusions(quantifiers, directions)
        return result
    
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

    def is_applicable(self, syllog, quantifiers):
        return True

def getProcesses() -> List[DirectionGenerator]:
    return [All(), FigureEffect(), MMT(), PHM()]