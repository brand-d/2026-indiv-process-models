''' Contains the CCOBRA model for TransSet, adapted for multiple choice.
    The model is based on the model provided here:
    https://github.com/brand-d/iccm-transset-indiv
'''
import ccobra
import numpy as np

# Constants
neg_quantifiers = ['E', 'O']
part_quantifiers = ['I', 'O']

def atmosphere_predictions(first, second):
    """ Produces atmosphere predictions to a given tuple of premises.

    Parameters
    ----------
    premises : list(str)
        List of premises (e.g., 'AA').

    Returns
    -------
    list(str)
        List of atmosphere predictions (e.g., ['Aac', 'Aca'])

    """
    premises = ''.join(sorted([first, second]))
    responses = []
    if premises == 'AA':
        responses = ['Aac', 'Aca']
    elif premises == 'AI':
        responses = ['Iac', 'Ica']
    elif premises == 'AE':
        responses = ['Eac', 'Eca']
    elif premises == 'AO':
        responses = ['Oac', 'Oca']
    elif premises == 'EE':
        responses = ['Eac', 'Eca']
    elif premises == 'EI':
        responses = ['Oac', 'Oca']
    elif premises == 'EO':
        responses = ['Oac', 'Oca']
    elif premises == 'II':
        responses = ['Iac', 'Ica']
    elif premises == 'IO':
        responses = ['Oac', 'Oca']
    elif premises == 'OO':
        responses = ['Oac', 'Oca']
    return responses

class TransSet(ccobra.CCobraModel):
    """ TransSet implementation.

    """

    def __init__(self, name='TransSet', individualized=True, save_params=True):
        """ Initializes the TransSet model.

        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the CCOBRA
            framework as a means for identifying the model.

        individualized : bool
            If true, the model adapts to an individual reasoner. Otherwise,
            default parameters are used (the model will behave as reported
            in the ICCM 2019 article 'On the Matter of Aggregate Models for
            Syllogistic Reasoning: A Transitive Set-Based Account for
            Predicting the Population')

        """
        super(TransSet, self).__init__(name, ['syllogistic'], ['single-choice', "multiple-choice"])

        self.individualized = individualized
        self.save_params = save_params

        # Parameters (default values as used in the 2019 version)
        self.nvc_aversion = 0.5 # 0 = None, 0.5 = Low, 1 = High
        self.anchor_set = "first" # first, most-recent
        self.particularity_rule = False
        self.negativity_rule = True

        self.best_fit = [self.nvc_aversion, self.anchor_set,
                        self.particularity_rule, self.negativity_rule]
        self.best_fits = [self.best_fit]

        # History (used for storing known responses of the current reasoner)
        self.history = []

    def pre_train_person(self, person_data):
        # When not using the individualized model, do not use pre_training
        if not self.individualized:
            return

        # get a list of parameter configurations best describing person_data
        self.best_fits = self.fit_to_history(person_data)

        # set the parameters of the model
        self.best_fit = self.best_fits[np.random.randint(len(self.best_fits))]
        self.nvc_aversion = self.best_fit[0]
        self.anchor_set = self.best_fit[1]
        self.particularity_rule = self.best_fit[2]
        self.negativity_rule = self.best_fit[3]

    def adapt(self, adapt_item, truth, **optionals):
        # When not using the individualized model, do not adapt
        if not self.individualized:
            return

        history_item = {
            "item": adapt_item,
            "response": truth
        }
        self.history.append(history_item)

        # retrain the model using the new information
        self.pre_train_person(self.history)

    def fit_to_history(self, history):
        if not self.individualized:
            return [[self.nvc_aversion,
                    self.anchor_set,
                    self.particularity_rule,
                    self.negativity_rule]]

        best_score = 0
        best_parameters = []

        for nvc_aversion in [0, 0.5, 1]:
            for anchor in ["first", "most-recent"]:
                for particularity_rule in [True, False]:
                    for negativity_rule in [True, False]:
                        score = 0

                        self.nvc_aversion = nvc_aversion
                        self.anchor_set = anchor
                        self.particularity_rule = particularity_rule
                        self.negativity_rule = negativity_rule

                        for task in history:
                            item = task["item"]
                            response = task["response"]
                            prediction = self.predict(item)
                            
                            if item.response_type == "multiple-choice":
                                string_target = set([ccobra.tuple_to_string(x) for x in response])
                                string_preds = set([ccobra.tuple_to_string(x) for x in prediction])
            
                                score += len(string_preds.intersection(string_target)) / len(string_preds.union(string_target))
                            elif prediction == response:
                                score += 1
                        if score == best_score:
                            best_score = score
                            best_parameters.append([
                                nvc_aversion, anchor,
                                particularity_rule,
                                negativity_rule])
                        elif score > best_score:
                            best_score = score
                            best_parameters = []
                            best_parameters.append([
                                nvc_aversion, anchor,
                                particularity_rule,
                                negativity_rule])
        return best_parameters

    def generate_prediction(self, figure, first, second):
        """ Generates predictions according the the TransSet theory.

        """

        # DETERMINE THE DIRECTION

        # Figure 3 and 4: There is no clear path to process a set of As to the
        # endpoint of Cs. Therefore, different heuristics are used to find a
        # direction. The quantifiers are ranked according to the ability to choose
        # a set with high confidence (therefore all > none > (some/somenot))
        ordering = { 'A' : 3, 'E' : 2, 'I' : 1, 'O' : 1}

        # The NVC aversion determines the participants likelyhood to respond with NVC.
        # For better interpretability and ease of fit, they are represented as one of
        # four values (none, low, moderate high). For easier thresholding, they will be
        # converted to numbers
        nvc_aversion = self.nvc_aversion

        # for figure 3 (where all paths point to B), the side with the 'more
        # informative' set is assumed to be the endpoint (e.g., "some A" and "all C", it is
        # reasonable to assume that the answer has to be a mapping from A to C, so
        # ac is the direction). For ties: NVC, as it is unclear which set is
        # a subset of the other
        if figure == '3':
            if ordering[second] > ordering[first]:
                figure = '1'
            elif ordering[second] < ordering[first]:
                figure = '2'
            else:
                # if the nvc aversion is to high, a direction has to be determined
                # This can be determined by a recency effect, or by relying on the
                # first object
                if nvc_aversion > 0.5:
                    figure = '1' if self.anchor_set == "first" else '2'
                else:
                    return "NVC"

        # figure 4 (all paths start from B) it is the other way round: the 'more
        # informative' set determines the starting point. As the premise starts with B, the
        # more informative set is a natural choice to be filtered by B, before the
        # second premise is applied.
        if figure == '4':
            if ordering[first] > ordering[second]:
                figure = '1'
            elif ordering[first] < ordering[second]:
                figure = '2'
            else:
                # if the nvc aversion is to high, a direction has to be determined
                # This can be determined by a recency effect, or by relying on the
                # first object
                if nvc_aversion > 0.5:
                    figure = '1' if self.anchor_set == "first" else '2'
                else:
                    return "NVC"

        dir = 'ac'
        # The direction in fig 1 ist ac, in fig 2 it is ca. Therefore the premise
        # order can also be changed
        if figure == '1':
            dir = 'ac'
        elif figure == '2':
            dir = 'ca'
            tmp = first
            first = second
            second = tmp
        else:
            print("Warning: figure should be 1 or 2, not {}".format(figure))

        # DETERMINE THE QUANTIFIER

        # It is assumed that the confidence in a path depends on the non-negative
        # quantifiers. The first premise is more important, as it is used in the
        # second premise to find the answer. therefore, a negative quantifier
        # in the first premise should increase the likelyhood of NVC more than a
        # negative quantifier in the second premise. Especially syllogisms with two
        # negative quantifiers will most likely be considered NVC. This might be
        # prevented by an NVC aversion though.
        if self.negativity_rule:
            if (first in neg_quantifiers) and nvc_aversion <= 0.5 and \
                not(second == 'A'):
                return "NVC"
            # if there is no NVC aversion at all, starting with a negation
            # can be enough to stop
            elif (first in neg_quantifiers) and nvc_aversion == 0:
                return "NVC"
            # if both quantifiers are negative, the nvc aversion has to be strong to
            # prevent NVC responses.
            elif (first in neg_quantifiers) and (second in neg_quantifiers):
                return "NVC"

        # Some people might use heuristics about the informativeness to determine
        # NVC. If there are two particular quantifiers, they might conclude NVC
        if (first in part_quantifiers) and (second in part_quantifiers) and \
            self.particularity_rule:
            return "NVC"

        # if both heuristics are present, a large bias towards NVC is possible
        # by using the partneg rule, when no NVC aversion is present
        if self.negativity_rule and self.particularity_rule:
            if (first != "A") and (second != "A") and nvc_aversion == 0:
                return "NVC"

        # After this pre-filtering, a combination heuristic is used for the rest
        # atmosphere is best compatible with the set based approach
        quantifier_prediction = atmosphere_predictions(first, second)

        # the figure determines the direction
        direction_idx = 0 if (dir == 'ac') else 1
        return quantifier_prediction[direction_idx]

    def predict(self, item, **kwargs):
        # Use CCOBRA to obtain the task encoding
        syl = ccobra.syllogistic.Syllogism(item)
        syllogism = syl.encoded_task

        # Generate the current prediction
        figure = syllogism[2]
        first = syllogism[0]
        second = syllogism[1]
        is_mc = item.response_type == "multiple-choice"
        pred = self.generate_prediction(figure, first, second)

        # return the decoded prediction
        if  is_mc:
            return [syl.decode_response(pred)]
        return syl.decode_response(pred)

    def end_participant(self, subj_id, model_log, **kwargs):
        if not self.save_params:
            return

        model_log["best_fits"] = self.best_fits

        model_log["nvc_aversion"] = self.best_fit[0]
        model_log["anchor_set"] = self.best_fit[1]
        model_log["particularity"] = self.best_fit[2]
        model_log["negativity"] = self.best_fit[3]
