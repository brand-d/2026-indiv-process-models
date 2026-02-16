
''' Contains the CCOBRA random baseline model for multiple-choice syllogisms.
Since the exact results of this model are not interpreted, no fixed seed is used.
'''
import numpy as np
import ccobra

class UniformModel(ccobra.CCobraModel):
    def __init__(self, name='UniformModel'):
        super(UniformModel, self).__init__(name, ["syllogistic"], ["single-choice", "multiple-choice"])

    def predict(self, item, **kwargs):
        syl = ccobra.syllogistic.Syllogism(item)
        if item.response_type == "single-choice":
            return item.choices[np.random.randint(len(item.choices))]
        results = []
        for choice in item.choices:
            if np.random.random() >= 0.5:
                results.append(choice)

        if len(results) == 0:
            results = ["NVC"]

        return results
