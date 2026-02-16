''' Contains the CCOBRA model for mReasoner for multiple choice.
    The model is taken and adapted from
    https://github.com/brand-d/cogsci-2023-multiplechoice
'''
import ccobra
import numpy as np

class MReasoner(ccobra.CCobraModel):
    def __init__(self, name='mReasoner'):
        super(MReasoner, self).__init__(name, ['syllogistic'], ['multiple-choice'])

        # Prepare cache
        self.cache = np.load('cache/nec_cache.npy')
        self.n_epsilon, self.n_lambda, self.n_omega, self.n_sigma = self.cache.shape[:-2]
        self.params = None

        # Normalize cache
        self.cache /= self.cache.sum(-1, keepdims=True)
        self.cache = np.nan_to_num(self.cache)

    def end_participant(self, identifier, model_log, **kwargs):
        paramnames = ['epsilon', 'lambda', 'omega', 'sigma']
        for pname, (_, pval) in zip(paramnames, self.params):
            model_log[pname] = pval

    def pre_train(self, dataset, **kwargs):
        tdata = np.zeros((64, 9))
        for subj_data in dataset:
            for task_data in subj_data:
                syl = ccobra.syllogistic.Syllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resps = [syl.encode_response(x) for x in task_data['response']]
                idx_task = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
                for enc_resp in enc_resps:
                    idx_resp = ccobra.syllogistic.RESPONSES.index(enc_resp)
                    tdata[idx_task, idx_resp] += 1 / len(enc_resps)

        # Perform fitting
        self.fit(tdata)

    def pre_train_person(self, data, **kwargs):
        self.pre_train([data])

    def get_score(self, pred, target):
        scores = []
        
        if np.sum(target) == 0:
            return 0
            
        if np.sum(pred) == 0:
            return 0
        
        for i in range(64):
            preds = set([ccobra.syllogistic.RESPONSES[x] for x in range(9) if pred[i, x] > 0.16])
            targets = set([ccobra.syllogistic.RESPONSES[x] for x in range(9) if target[i, x] > 0.16])
            scores.append(len(preds.intersection(targets)) / len(preds.union(targets)))
        return np.mean(scores)

    def fit(self, tdata):
        best_score = -1
        best_params = None
        
        # Iterate over parameterizations in the cache
        for idx_epsilon, p_epsilon in enumerate(np.linspace(0, 1, self.n_epsilon)):
            for idx_lambda, p_lambda in enumerate(np.linspace(0.1, 8, self.n_lambda)):
                for idx_omega, p_omega in enumerate(np.linspace(0, 1, self.n_omega)):
                    for idx_sigma, p_sigma in enumerate(np.linspace(0, 1, self.n_sigma)):
                        params = (idx_epsilon, idx_lambda, idx_omega, idx_sigma)
                        cache_mat = self.cache[params]
                        if np.sum(cache_mat) == 0:
                            continue

                        # Compare cache with training data
                        score = self.get_score(cache_mat, tdata)
        
                        if score > best_score:
                            best_score = score
                            best_params = list(zip(params, (p_epsilon, p_lambda, p_omega, p_sigma)))
        
        # Set to best params
        self.params = best_params

    def predict(self, item, **kwargs):
        # Obtain task information
        syl = ccobra.syllogistic.Syllogism(item)
        enc_task = syl.encoded_task
        idx_task = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
        
        # Obtain prediction matrix
        param_idxs = tuple(x[0] for x in self.params)
        cache_mat = self.cache[param_idxs]
        cache_pred = cache_mat[idx_task]
        
        responses = []
        for resp_idx, resp in enumerate(ccobra.syllogistic.RESPONSES):
            if cache_pred[resp_idx] >= 0.16:
                responses.append(resp)

        # Generate prediction
        return [syl.decode_response(x) for x in responses]
