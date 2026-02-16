''' CCOBRA implementation of the "Automated process generation" model
    using a greedy fitting process. The model combines sub-process components
    to a process model fitted individually to a participant.
'''
import numpy as np
import ccobra
from typing import Dict, List, Optional, Union, Collection, cast

import actions

def metric(observation: Union[str, Collection[str]], prediction: Collection[str]) -> float:
    ''' Realizes the metric used to optimize (given observations O and predictions P).
    For multiple-choice (as in the paper), it is the Jaccard coefficient:
    jacc(O, P) = |O intersect P| / |O union P|

    For single-choice, it is the accuracy, given a list of predictions:
    acc(O, P) = (1 if O in P, else 0) / |P|

    Parameters
    ----------
    observation: Union[str, Collection[str]]
        Observed response, either a string only (single choice) or a collection of conclusion-strings (multiple choice)

    prediction: Collection[str]
        Collection of conclusion-strings of the prediction
    
    Returns
    -------
    float
        Score according to the respective metric
        
    '''
    if isinstance(observation, str):
        # In the case of single-choice, use accuracy/hitrate
        return (observation in prediction) / len(prediction)
    else:
        # In the case of multiple choice, use Jaccard coefficient
        obs_set = set(observation)
        pred_set = set(prediction)
        return len(obs_set.intersection(pred_set)) / len(obs_set.union(pred_set))

class ProcessModel(ccobra.CCobraModel):
    def __init__(self, name='ProcessModel'):
        super(ProcessModel, self).__init__(name, ["syllogistic"], ["single-choice", "multiple-choice"])
        
        # Initialize with Atmosphere and Figure Effect as a default
        self.conversion : actions.conversions.Conversion = actions.conversions.Identity()
        self.nvc_rules : Collection[actions.nvc_rules.NVCDirectRule] = []
        self.quantifier_gen : actions.quantifiers.QuantifierGenerator = actions.quantifiers.Atmosphere()
        self.direction_gen : actions.directions.DirectionGenerator = actions.directions.FigureEffect()
        self.extensions : Collection[actions.extensions.Extension] = []
        self.conclusion_tests : Collection[actions.conclusion_tests.ConclusionTest] = []
        self.preferences : Collection[actions.preferences.Preference] = []
        self.is_fol : bool = False
    
    def _predict_with_pipeline(self,
                               syllog: str, 
                               conversion: actions.conversions.Conversion, 
                               nvc_rules: Collection[actions.nvc_rules.NVCDirectRule], 
                               quantifier: actions.quantifiers.QuantifierGenerator, 
                               direction: actions.directions.DirectionGenerator, 
                               extensions: Collection[actions.extensions.Extension], 
                               conclusion_tests: Collection[actions.conclusion_tests.ConclusionTest], 
                               preferences: Collection[actions.preferences.Preference]) -> List[str]:
        ''' Generates a list of selected conclusions with a given pipeline configuration.
        The pipeline corresponds directly to the pipeline illustrated in the paper.

        Parameters
        ----------
        syllog : str
            The syllogism to make predictions for (in abbreviated form, e.g., IO3)
        
        conversion: actions.Conversions.Conversion
            The conversion to use (Path, Illicit, Transitive, or None (realized as Identity))
        
        nvc_rules: Collection[actions.NVCDirect.NVCDirectRule]
            A collection of the NVC rules to use for prediction

        quantifier: actions.Quantifiers.QuantifierGenerator,
            The quantifier generation mechanism to use
        
        direction: actions.Directions.DirectionGenerator, 
            The mechanism to determine the direction of the initial set of partial conclusions
        
        extensions: Collection[actions.Extensions.Extension], 
            A collection of the extension mechanisms to use
        
        conclusion_tests: Collection[actions.ConclusionTests.ConclusionTest], 
            A collection of the conclusion tests to use
        
        preferences: Collection[actions.Preferences.Preference]
            A collection of the preference mechanisms to use
            
        Returns
        -------
        List[str]
            A list of selected conclusions based on the pipeline

        '''
        # Apply conversion
        if conversion and conversion.is_applicable(syllog):
            syllog = conversion.get_conversion(syllog)
        
        # Directly filter by NVC rules
        for nvc_direct in nvc_rules:
            if nvc_direct.is_nvc(syllog):
                return ["NVC"]
        
        # Generate quantifiers
        quants = ["A", "I", "E", "O"]
        if quantifier.is_applicable(syllog):
            quants = quantifier.get_quantifier(syllog)
        
        # Generate directions
        conclusions = set()
        if direction.is_applicable(syllog, quants):
            conclusions = direction.get_directions(syllog, quants)
        else:
            directions = ["ac", "ca"]
            for q in quants:
                for d in directions:
                    conclusions.add("{}{}".format(q, d))
        
        # Extend conclusions
        for extension in extensions:
            conclusions = extension.extend(syllog, conclusions)
        conclusions = list(conclusions)
 
        # Check for NVC
        for concl_test in conclusion_tests:
            conclusions = concl_test.test(syllog, conclusions)
        conclusions = [x for x in conclusions if x != "NVC"]
        if not conclusions:
            return ["NVC"]

        # Preference results
        for p in preferences:
            conclusions = p.select(syllog, conclusions)
        
        return conclusions
    
    def _predict_fol(self, syllog: str, preferences: Collection[actions.preferences.Preference]) -> List[str]:
        ''' Predicts the conclusions based on first order logic, but 
        can consider preferences. Therefore, a given response will be potentially
        miss some valid conclusions, but not contain conclusions that do not
        logically follow from the premises.

        Parameters
        ----------
        syllog : str
            The syllogism to make predictions for (in abbreviated form, e.g., IO3)
        
        preferences: Collection[actions.Preferences.Preference]
            A collection of the preference mechanisms to use
            
        Returns
        -------
        List[str]
            A list of selected conclusions based on the pipeline
        '''
        # obtain conclusions
        conclusions = ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES[syllog]
        
        # Preference results
        for p in preferences:
            conclusions = p.select(syllog, conclusions)
    
        return conclusions

    def predict(self, item, **kwargs):
        ''' CCOBRA predict function
        '''
        # Process syllogism
        syllogism = ccobra.syllogistic.Syllogism(item)
        enc_task = syllogism.encoded_task
        is_mc = item.response_type == "multiple-choice"
        
        predictions = ["NVC"]

        # If first-order logic should be used for the respective participant,
        # bypass the pipeline (see explanation in the paper under "Fitting Process")
        if self.is_fol:
            predictions = self._predict_fol(enc_task, self.preferences)
        # Otherwise, use the pipeline with the best configuration found in training
        else:
            predictions = self._predict_with_pipeline(enc_task, 
                                                    self.conversion,
                                                    self.nvc_rules,
                                                    self.quantifier_gen,
                                                    self.direction_gen,
                                                    self.extensions,
                                                    self.conclusion_tests,
                                                    self.preferences)
        if is_mc:
            return [syllogism.decode_response(x) for x in predictions]
        else:
            # Only for single choice, it may be necessary to chose between responses,
            # in case the pipeline didn't yield a unique one, for multiple choice 
            prediction = np.random.choice(predictions)
            return syllogism.decode_response(prediction)

    def pre_train_person(self, person_data: List[Dict[str, object]]) -> None:
        ''' Realizes the fitting process of the pipeline.
        Thereby, it is done as follows:
        1. It is tested how well the participant is accounted for using FOL
        2. Fitting is done for the "generative" parts that creeate conclusion candidates:
            - Conversion, Quantifier, Direction create the first conclusion candidates
            - Extensions increase it to account for entailments, etc
            - Extensions are selected greedily together with preliminary selections of preferences
        3. Reducing components are determined:
            - NVC rules directly overwrite any conclusion
            - Conclusion tests and preferences reduce the set of conclusions
        
        The rationale behind the steps are described in the paper under "Fitting process"

        Parameters
        ----------
        person_data : List[Dict[str, object]]
            List with observations of the participant to fit
        '''
        # Variables for keeping track of the progress
        best_acc_fol = 0
        best_prefs_fol : Collection[actions.preferences.Preference] = []

        # 1. FOL performance: Iterate over the possible combinations of preferences
        for preferences in actions.powerset(actions.preferences.getProcesses()):
            preferences = cast(Collection[actions.preferences.Preference], preferences)

            # Calculate the overall performance with the respective preferences over all tasks
            acc = 0
            for task in person_data:
                item: ccobra.Item = task["item"]
                syllogism = ccobra.syllogistic.Syllogism(item)
                is_mc = item.response_type == "multiple-choice"

                response = None
                pred = set(self._predict_fol(syllogism.encoded_task, preferences))
                if not is_mc:
                    response = syllogism.encode_response(task["response"])
                else:
                    response = set([syllogism.encode_response(x) for x in task["response"]]) # type: ignore

                score = metric(response, pred)
                acc += score
            acc /= len(person_data)

            # Store the current best configuration
            if acc > best_acc_fol:
                best_acc_fol = acc
                best_prefs_fol = preferences

        print("FOL performance:", best_acc_fol)
        print("    Preferences:", [x.name for x in best_prefs_fol])
        print()
        
        # Start search for configuration if not FOL
        print("Finding process model")
        best_acc_proc = 0

        # Create a dictionary that maps the syllogisms to the respective responses (observations)
        truth_dict = {}
        for task in person_data:
            item = task["item"]
            syllogism = ccobra.syllogistic.Syllogism(item)
            is_mc = item.response_type == "multiple-choice"
            response = None
            if is_mc:
                response = set([syllogism.encode_response(x) for x in task["response"]]) # type: ignore
            else:
                response = syllogism.encode_response(task["response"])
            truth_dict[syllogism.encoded_task] = response

        # 2. Find maximum conclusion set (Conversions, Quantifier, Direction, Extensions)
        print("    Finding model for syllogs with non-NVC responses")
        for conversion in actions.conversions.getProcesses():
            for quant in actions.quantifiers.getProcesses():
                for direction in actions.directions.getProcesses():
                    prediction_dict = {}
                    
                    # For the current generative phase (+ Conversion), generate predictions for all syllogs
                    # This represents the results after the first two phases in the pipeline: With a first full set of conclusion candidates
                    for syllog in ccobra.syllogistic.SYLLOGISMS:
                        pred = self._predict_with_pipeline(syllog, 
                                                conversion,
                                                [],
                                                quant,
                                                direction,
                                                [],
                                                [],
                                                [])
                        prediction_dict[syllog] = pred

                    # Start searching for a combination of extensions in a greedy fashion:
                    # Iteratively choose the best extension (temporarily together with a preference)
                    # Preferences are temporarily included to allow to counterbalance the "inflation"
                    # and overshooting done by extensions.
                    optim_ext = []
                    optim_prefs = []
                    acc_pref = self._get_dict_score(truth_dict, prediction_dict)
                    acc_ext = acc_pref

                    ext_list = [x for x in actions.extensions.getProcesses()]
                    pref_list = [x for x in actions.preferences.getProcesses()]
                    improved = True

                    # The greedy search for extensions is done until no improvement was possible
                    while improved:
                        improved = False
                        start_pref = acc_pref
                        start_ext = acc_ext
                        
                        # Improve extensions: Select the next best
                        ext_to_add = []
                        for ext in ext_list:
                            cur_ext = [ext] + optim_ext

                            # Calculate score with the added extension
                            cur_acc_ext = self._get_dict_score(truth_dict, prediction_dict, prefs=optim_prefs, ext=cur_ext)

                            # Store added extension if better, to add it if no better is found
                            if cur_acc_ext > acc_ext:
                                ext_to_add = [ext]
                                acc_ext = cur_acc_ext
                        # Add the new extension to the list
                        optim_ext.extend(ext_to_add)
                        if ext_to_add:
                            ext_list.remove(ext_to_add[0])

                        # Store that improvement was made
                        if cur_acc_ext > start_ext:
                            improved = True

                        # Improve preferences (temporarily) to balance the extensions
                        pref_to_add = []
                        for pref in pref_list:
                            cur_pref = [pref] + optim_prefs
                            # Calculate score with the added preference
                            cur_acc_pref = self._get_dict_score(truth_dict, prediction_dict, ext=optim_ext, prefs=cur_pref)

                            # Store added preference if better, to add it if no better is found
                            if cur_acc_pref > acc_pref:
                                pref_to_add = [pref]
                                acc_pref = cur_acc_pref
                        # Add the new preference to the list
                        optim_prefs.extend(pref_to_add)
                        if pref_to_add:
                            pref_list.remove(pref_to_add[0])

                        # Store that improvement was made
                        if acc_pref > start_pref:
                            improved = True
              
                    # Calculate the score for the current configuration
                    acc = self._get_score(person_data,
                                        conversion, 
                                        [], 
                                        quant, 
                                        direction, 
                                        optim_ext, 
                                        [], 
                                        optim_prefs)
                    # Store the currently best configuration
                    # Note that preferences are not yet final, the others are
                    if acc > best_acc_proc:
                        self.conversion = conversion
                        self.quantifier_gen = quant
                        self.direction_gen = direction
                        self.extensions = optim_ext
                        self.preferences = optim_prefs
                        best_acc_proc = acc

        print("        Selected Quantifier: {}".format(self.quantifier_gen.name))
        print("        Selected Direction:  {}".format(self.direction_gen.name))
        print("        Selected Conversion: {}".format(self.conversion.name))
        print("        Selected Extensions: {}".format([x.name for x in self.extensions]))
        print("        Intermediate Prefs.:    {}".format([x.name for x in self.preferences]))
        print("    Intermediate Acc: ", best_acc_proc)
        print()

        # Find the set of NVC rules
        # In the cognitive interpretation, these rules apply ahead of the others,
        # but they are independent and overwrite the others. 
        # Therefore, they can be selected indepently, reducing search complexity,
        # while having little impact on performance. As indicated in the paper,
        # only the interaction with conversion is neglected in doing so,
        # potentially reducing optimality by a small amount.
        # However, interpretation wise, it is unintuitive to select a task interpretation
        # only to then directly deem the task "invalid".
        print("    Finding NVC rules")
        for nvc_direct in actions.powerset(actions.nvc_rules.getProcesses()):
            nvc_direct = cast(Collection[actions.nvc_rules.NVCDirectRule], nvc_direct)
            acc = self._get_score(person_data,
                                self.conversion, 
                                nvc_direct, 
                                self.quantifier_gen, 
                                self.direction_gen, 
                                self.extensions, 
                                [], 
                                self.preferences)
            if acc > best_acc_proc:
                best_acc_proc = acc
                self.nvc_rules = cast(Collection[actions.nvc_rules.NVCDirectRule], nvc_direct)

        # The remaining two steps in the pipeline only reduce the candidate set,
        # and can thereby be optimized together
        print("    Finding Conclusion Tests and Preferences")
        for conclusion_test in actions.conclusion_tests.get_process_combinations():
            best_pref = []
            best_pref_score = self._get_score(person_data,
                                        self.conversion, 
                                        self.nvc_rules, 
                                        self.quantifier_gen, 
                                        self.direction_gen, 
                                        self.extensions, 
                                        conclusion_test, 
                                        [])
            prefs_to_test = [x for x in actions.preferences.getProcesses()]
            improved = True
            while improved:
                improved = False
                # Select the next best preference greedily,
                # similar to the extensions before
                best_pref_this_round = None
                for pref in prefs_to_test:
                    test_prefs = best_pref + [pref]
                    score_with_pref = self._get_score(person_data,
                                        self.conversion, 
                                        self.nvc_rules, 
                                        self.quantifier_gen, 
                                        self.direction_gen, 
                                        self.extensions, 
                                        conclusion_test, 
                                        test_prefs)
                    if score_with_pref > best_pref_score:
                        best_pref_this_round = pref
                        best_pref_score = score_with_pref
                if best_pref_this_round:
                    best_pref.append(best_pref_this_round)
                    improved = True
                    prefs_to_test.remove(best_pref_this_round)

            # Store the best identified configuration
            if best_pref_score > best_acc_proc:
                best_acc_proc = best_pref_score
                self.conclusion_tests = conclusion_test
                self.preferences = best_pref
            
        print("        Selected NVC Direct:  {}".format([x.name for x in self.nvc_rules]))
        print("        Selected Conclusion Tests:   {}".format([x.name for x in self.conclusion_tests]))
        print("        Selected Preferences: {}".format([x.name for x in self.preferences]))
        print("    Final Acc: ", best_acc_proc)
        print()
        
        # Compare FOL results with the pipeline configuration and decide accordingly
        if best_acc_fol > best_acc_proc:
            self.is_fol = True
            self.preferences = best_prefs_fol
            self.fit_score = best_acc_fol
        else:
            self.is_fol = False
            self.fit_score = best_acc_proc
    
    def _get_dict_score(self, 
                        observations: Dict[str, Union[str, Collection[str]]], 
                        pred_dict: Dict[str, Collection[str]], 
                        ext: Optional[Collection[actions.extensions.Extension]]=None, 
                        prefs: Optional[Collection[actions.preferences.Preference]]=None) -> float:
        ''' Calculates the score given a prediction dictionary and a dictionary of observations.
        Thereby, preferences and extensions can be given as well. That way, the function
        can calculate the score quicker than using a full prediction (using _get_score).
        Essentially, extensions and preferences can be applied to precomputed sets of conclusions.

        Parameters
        ----------
        observations: Dict[str, Union[str, Collection[str]]]
            Observation dictionary, mapping syllogisms to the respective observed conclusions.
        
        pred_dict: Dict[str, Collection[str]]
            Prediction dictionary, mapping syllogisms to the respective observed conclusions.
        
        ext: Collection[actions.Extensions.Extension] | None
            Optionally, a collection of extensions can be given to be applied to the predictions
        
        prefs: Collection[actions.Preferences.Preference] | None
            Optionally, a collection of preferences can be given to be applied to the predictions
            
        Returns
        -------
        float
            The score when using the provided observations and predictions. Uses the global metric function.

        '''
        acc = 0
        total = 0
        if not ext:
            ext = []
        if not prefs:
            prefs = []

        for syllog in ccobra.syllogistic.SYLLOGISMS:
            if syllog not in observations:
                continue
            preds = pred_dict[syllog]
            for ex in ext:
                preds = ex.extend(syllog, preds)
            for pref in prefs:
                preds = pref.select(syllog, preds)
            acc += metric(observations[syllog], preds)
            total += 1
        return acc/total

    def _get_score(self, 
                   observations: Collection[Dict[str, object]], 
                   conversion: actions.conversions.Conversion, 
                   nvc_rules: Collection[actions.nvc_rules.NVCDirectRule], 
                   quantifier: actions.quantifiers.QuantifierGenerator, 
                   direction: actions.directions.DirectionGenerator, 
                   extensions: Collection[actions.extensions.Extension], 
                   concl_tests: Collection[actions.conclusion_tests.ConclusionTest], 
                   preferences: Collection[actions.preferences.Preference]) -> float:
        ''' Calculates the score/metric for a given set of observations and a complete specification
        of the pipeline configuration. This uses a full pipeline-prediction, and is therefore slower
        than the precomputed "_get_dict_score", which can be used for extensions and preferences, only.

        Parameters
        ----------
        observations: Collection[Dict[str, object]]
            Observation collection, containing tasks and respective responses
        
        conversion: actions.Conversions.Conversion
            The conversion to use (Path, Illicit, Transitive, or None (realized as Identity))
        
        nvc_rules: Collection[actions.NVCDirect.NVCDirectRule]
            A collection of the NVC rules to use for prediction

        quantifier: actions.Quantifiers.QuantifierGenerator,
            The quantifier generation mechanism to use
        
        direction: actions.Directions.DirectionGenerator, 
            The mechanism to determine the direction of the initial set of partial conclusions
        
        extensions: Collection[actions.Extensions.Extension], 
            A collection of the extension mechanisms to use
        
        conclusion_tests: Collection[actions.ConclusionTests.ConclusionTest], 
            A collection of the conclusion tests to use
        
        preferences: Collection[actions.Preferences.Preference]
            A collection of the preference mechanisms to use

        Returns
        -------
        float
            The score when using the provided observations and pipeline configuration. Uses the global metric function.

        '''
        acc = 0
        for task in observations:
            item: ccobra.Item = task["item"]
            syllogism = ccobra.syllogistic.Syllogism(item)
            is_mc = item.response_type == "multiple-choice"
            response = None
            if is_mc:
                response = [syllogism.encode_response(x) for x in task["response"]] # type: ignore
            else:
                response = syllogism.encode_response(task["response"])
            pred = self._predict_with_pipeline(syllogism.encoded_task, 
                                                conversion,
                                                nvc_rules,
                                                quantifier,
                                                direction,
                                                extensions,
                                                concl_tests,
                                                preferences)
            score = metric(response, pred)
            acc += score
        return acc / len(observations)

    def end_participant(self, subj_id, model_log, **kwargs):
        ''' Implements the respective CCOBRA function, to allow saving the parameter configuration
        in a dictionary. This can later be accessed in the benchmark file.
        '''
        model_log["use_fol"] = self.is_fol

        if not self.is_fol:
            model_log["quantifier"] = self.quantifier_gen.name
            model_log["direction"] = self.direction_gen.name
            model_log["conversion"] = self.conversion.name
            model_log["extensions"] = [x.name for x in self.extensions]
            model_log["nvc_direct"] = [x.name for x in self.nvc_rules]
            model_log["test_concl"] = [x.name for x in self.conclusion_tests]
            
        model_log["preferences"] = [x.name for x in self.preferences]
        
        print("Finished participant", subj_id)
        print("-------------------------")
        print()
