''' Conversion (sub-)processes.
    Realizes a function: syllog --> syllog

    In the paper, those are part of the first phase (Task Interpretation),
    and represent the respective stage in the pipeline (see Figure 1).
'''
from .. import base
from typing import List

class Conversion(base.CognitiveSubprocess):
    def __init__(self, name):
        super(Conversion, self).__init__(name)
    
    def get_name(self):
        return "Conversion.{}".format(self.name)

    def apply(self, state_info):
        if not self.is_applicable(state_info["task"]):
            return state_info
        new_task = self.get_conversion(state_info["task"])
        return {
            "phase": base.Phase.TASK_INTERPRETATION,
            "task": new_task,
            "state": new_task
        }

    def get_conversion(self, syllog):
        raise NotImplementedError()
    
    def is_applicable(self, syllog):
        raise NotImplementedError()
    
    def is_applicable_in_phase(self, phase):
        return phase == base.Phase.TASK_INTERPRETATION

class Identity(Conversion):
    ''' Pseudo-process equivalent to not using a conversion.
    '''
    def __init__(self):
        super(Identity, self).__init__("Identity")
    
    def get_conversion(self, syllog):
        return syllog
    
    def is_applicable(self, syllog):
        return True

class TransSet(Conversion):
    ''' Conversion realizing the mechanism of TransSet to find a
        transitive path.

        For more information, refer to the introduction in the paper or
        Brand, et al. (2020). Extending TransSet: An Individualized Model for Human Syllogistic Reasoning.
    '''
    def __init__(self):
        super(TransSet, self).__init__("TransSet")
        self.ordering = { 'A' : 3, 'E' : 2, 'I' : 1, 'O' : 1}

    def is_applicable(self, syllog):
        figure = syllog[2]
        q1 = syllog[0]
        q2 = syllog[1]
        if (figure == "3") or (figure == "4"):
            return self.ordering[q2] != self.ordering[q1]

    def get_conversion(self, syllog):
        figure = syllog[2]
        q1 = syllog[0]
        q2 = syllog[1]
        if figure == "1":
            return syllog
        elif figure == "2":
            return syllog
        elif figure == "3":
            if self.ordering[q2] > self.ordering[q1]:
                return "{}{}1".format(q1, q2)
            elif self.ordering[q2] < self.ordering[q1]:
                return "{}{}2".format(q1, q2)
            else:
                return syllog
        elif figure == "4":
            if self.ordering[q1] > self.ordering[q2]:
                return "{}{}1".format(q1, q2)
            elif self.ordering[q1] < self.ordering[q2]:
                return "{}{}2".format(q1, q2)
            else:
                return syllog

class IllicitConversion(Conversion):
    ''' Assumes that people to the illicit conversions to obtain an easier direction:
    3 or 4 --> 1 or 2
    Note that this is an interpretation of the illicit conversion hypothesis,
    which only states that conversions with A and O are considered.

    Inspired by the Illicit Conversion Hypothesis, see
    Revlis, (1975). Two models of syllogistic reasoning: Feature selection and conversion.

    We implemented it as a special case of the normal transistive conversion.
    '''
    def __init__(self):
        super(IllicitConversion, self).__init__("IllicitConversion")
        self.inner_conversion = TransitiveConversion(allowed_quants=["A", "E", "I", "O"])
    
    def is_applicable(self, syllog):
        return self.inner_conversion.is_applicable(syllog)

    def get_conversion(self, syllog):
        return self.inner_conversion.get_conversion(syllog)

class TransitiveConversion(Conversion):
    ''' Assumes that people to a conversions to obtain an easier direction:
    Figure 3 or 4 --> 1 or 2
    
    It realizes a logically valid conversion, since only E and I are reversed.

    For motivation, refer to the paper.
    '''
    def __init__(self, allowed_quants=None):
        super(TransitiveConversion, self).__init__("TransitiveConversion")
        if not allowed_quants:
            allowed_quants = ["E", "I"]
        self.allowed_quants = allowed_quants
    
    def is_applicable(self, syllog):
        figure = syllog[2]
        q1 = syllog[0]
        q2 = syllog[1]
        if figure == "1":
            return False # Nothing to do
        elif figure == 2:
            return False # Nothing to do
        else:
            # Any switch is good
            return (q1 in self.allowed_quants) or (q2 in self.allowed_quants)

    def get_conversion(self, syllog):
        figure = syllog[2]
        q1 = syllog[0]
        q2 = syllog[1]
        self.allowed_quants = ["A", "O"]
        if figure == "1":
            return syllog
        elif figure == "2":
            return syllog
        elif figure == "3":
            # Prefer to generate figure 1
            if (q2 in self.allowed_quants):
                return "{}{}1".format(q1, q2)
            elif (q1 in self.allowed_quants):
                return "{}{}2".format(q1, q2)
            else:
                return syllog
        else:
            # Prefer to generate figure 1
            if (q1 in self.allowed_quants):
                return "{}{}1".format(q1, q2)
            elif (q2 in self.allowed_quants):
                return "{}{}2".format(q1, q2)
            else:
                return syllog

def getProcesses() -> List[Conversion]:
    return [Identity(), TransitiveConversion(), TransSet(), IllicitConversion()]