from itertools import chain, combinations
from typing import Iterator, Collection
from .base import CognitiveSubprocess

def powerset(input_list : Collection[CognitiveSubprocess]) -> Iterator[Collection[CognitiveSubprocess]]:
    ''' Helper function generating a powerset of a given list.

    Parameters
    ----------

    input_list : Collection[CognitiveSubprocess]
        The input collection of cognitive subprocesses.

    Returns
    -------
    Iterator[Collection[CognitiveSubprocess]]
        An iterator that gives out each combination in the powerset of the input list once at a time.

    '''
    return chain.from_iterable(combinations(input_list, r) for r in range(len(input_list)+1))