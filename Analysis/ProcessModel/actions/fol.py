''' FOL subprocess that answers using first-order logic.
    Directly generates a conclusion set from a task
    Realizes a function: syllog --> set(conclusions)
'''
from .base import CognitiveSubprocess, Phase
import ccobra

class FOLSubprocess(CognitiveSubprocess):
    def __init__(self):
        super(FOLSubprocess, self).__init__("FOL")
    
    def get_name(self):
        return "FOL"

    def is_applicable_in_phase(self, phase):
        return phase == Phase.TASK_INTERPRETATION
    
    def apply(self, state_info):
        return {
            "phase": Phase.CANDIDATE_ALTERATION,
            "task": state_info["task"],
            "state": ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES[state_info["state"]]
        }