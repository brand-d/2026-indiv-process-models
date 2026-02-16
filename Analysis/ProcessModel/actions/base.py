from enum import Enum
from typing import Dict, Collection

class Phase(Enum):
    TASK_INTERPRETATION = 1
    CANDIDATE_GENERATION = 2
    CANDIDATE_ALTERATION = 3

class CognitiveSubprocess:
    ''' Base class for Cognitive Subprocesses / Actions.
    Those can only be applied in their respective phase.
    '''
    def __init__(self, name):
        self.name = name
    
    def is_applicable_in_phase(self, phase: Phase) -> bool:
        ''' Determines if the action/Subprocess is applicable in the current phase.

        Parameters
        ----------
        phase: Phase
            The current phase

        Returns
        -------
        bool
            True, iff applicable
        '''
        raise NotImplementedError()
    
    def apply(self, state_info: Dict[str, Phase | str | Collection[str]]) -> Dict[str, str | Collection[str]]:
        ''' Applies the subprocess/action to the current state.

        Parameters
        ----------
        state_info: Dict[str,  Phase | str | Collection[str]]
            The current state, as a dictionary. The dictionary contains phase, task, and current conclusion candidates.

        Returns
        -------
        Dict[str, object]
            The state after the subprocess/action is applied.
        '''
        raise NotImplementedError()
    
    def get_name(self) -> str:
        ''' Returns the name of the action/subprocess.

        Returns
        -------
        str
            The name
        '''
        raise NotImplementedError()
