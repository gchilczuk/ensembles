from collections import Counter

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from datasets import Data


def test_meta_param(classifier, data: Data, c_params, meta_tested, meta_values, tested_param, tested_values):
    scores = {}
    for mv in meta_values:
        c_params[meta_tested] = mv
        result = test_param(classifier, data, c_params, tested_param, tested_values, s=('f1_micro',))
        scores[mv] = result['F1-score']
    return scores


def test_dectree_meta(classifier, data: Data, c_params, meta_tested, meta_values, tested_param, tested_values):
    scores = {}
    for mv in meta_values:
        c_params[meta_tested] = mv
        result = test_param(classifier, data, c_params, tested_param, tested_values, s=('f1_micro',))
        scores[f'deepth={mv.max_depth}'] = result['F1-score']
    return scores


def test_param(classifier, data: Data, c_params, tested_param, tested_values, s=None, sn=None):
    score_keys = s or ('f1_micro', 'precision_micro', 'recall_micro', 'accuracy')
    sn = sn or ('F1-score', 'precision', 'recall', 'accuracy')
    scores = {}
    models = [classifier(**c_params, **{tested_param: tv}) for tv in tested_values]
    for score, score_name in zip(score_keys, sn):
        results = {}
        for model, tv in zip(models, tested_values):
            splits = min(min(Counter(data.y).values()), 10)
            res = cross_val_score(
                model, data.x, data.y,
                cv=StratifiedKFold(n_splits=splits, shuffle=True),
                scoring=score, error_score=np.nan,
            )
            results[tv] = np.average(res)

        scores[score_name] = results
    return scores
