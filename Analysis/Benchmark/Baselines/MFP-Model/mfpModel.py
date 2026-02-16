''' Contains the CCOBRA model for the "Most frequent Pattern" baseline.
    The model responds with the combination of responses most frequent in the data,
    thereby being the multiple-choice equivalent to the most-frequent answer baseline.

    Code is based on Brand et Al. 2023 (https://github.com/brand-d/cogsci-2023-multiplechoice)
'''
import ccobra
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

def get_mfp_dict(dataset: List[List[Dict[str, object]]]) -> Dict[str, List[Tuple[Tuple[str], int]]]:
    ''' Generates a dictionary mapping sylogisms to a list of response tuples 
    (containing the conclusions selected together) with the respective frequencies).

    Parameters
    ----------
    dataset : List[List[Dict[str, object]]]
        The dataset containing other participants with all their tasks and responses
    
    Returns
    -------
    Dict[str, List[Tuple[Tuple[str], int]]]
        Dictionary with the most frequently selected pattern(s) for each syllogism
    '''
    mfp = {}
    for subj_train_data in dataset:
        for seq_train_data in subj_train_data:
            syllog = ccobra.syllogistic.Syllogism(seq_train_data["item"])
            responses = [syllog.encode_response(x) for x in seq_train_data["response"]] # type: ignore
            syl = syllog.encoded_task
            
            if syl not in mfp:
                mfp[syl] = []
            mfp[syl].append(sorted(responses))

    mfp_res = {}
    for key, value in mfp.items():
        value = np.array([tuple(x) for x in value], dtype=tuple)

        numbers = dict(zip(*np.unique(value, return_counts=True)))
        numbers = sorted(numbers.items(), key=lambda x: x[1], reverse=True)

        mfp_res[key] = [numbers[0]]
        
        # Add all patterns that have the equal score to the best, too
        for i in range(1, len(numbers)):
            if numbers[0][1] == numbers[i][1]:
                mfp_res[key].append(numbers[i])
            else:
                break

    return mfp_res

class MFPModel(ccobra.CCobraModel):
    ''' Model responding with the most frequently selected pattern.
    '''
    def __init__(self, name='MFP', response_csv=None):
        super(MFPModel, self).__init__(name, ['syllogistic'], ['multiple-choice'])

    def pre_train(self, dataset):
        # Obtain most-frequent pattern dictionary
        mfp_dict = get_mfp_dict(dataset)
        mfp_storage = {}

        # Only store the patterns
        for syl, mfps in mfp_dict.items():
            syl_patterns = []
            for mfp in mfps:
                syl_patterns.append([x for x in mfp[0]])
            mfp_storage[syl] = syl_patterns
        self.mfp_dict = mfp_storage

    def predict(self, item, **kwargs):
            syl = ccobra.syllogistic.Syllogism(item)
            enc_task = syl.encoded_task

            # Retrieve predictions from dictionary
            preds = self.mfp_dict[enc_task]

            # In cases multiple exactly equal frequent patterns exist, select one at random
            pred = preds[np.random.randint(len(preds))]
            return [syl.decode_response(x) for x in pred]
