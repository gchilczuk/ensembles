from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from datasets import Data
from experiments import test_param
from plotter import scores_dict_table, flatplot

c_params = {
    'random_state': 1,
    'n_estimators': 25
}


def test_criterion(data: Data):
    return test_param(RandomForestClassifier, data, c_params, 'criterion', ('gini', 'entropy'))


def test_bootstrap(data: Data):
    return test_param(RandomForestClassifier, data, c_params, 'bootstrap', (True, False))


def test_max_depth(data: Data):
    return test_param(RandomForestClassifier, data, c_params, 'max_depth', (1, 3, 8, 13, 21, None))


def test_min_samples_split(data: Data):
    return test_param(RandomForestClassifier, data, c_params, 'min_samples_split', (2, 3, 8, 13, 21, 34))


def test_n_estimators(data: Data):
    c_params = {
        'random_state': 1,
    }
    return test_param(RandomForestClassifier, data, c_params, 'n_estimators', np.arange(1, 205, 3), s=('f1_micro',))


def run(dss: List[Data]):
    for test in [test_criterion, test_bootstrap, test_max_depth, test_min_samples_split]:
        for d in dss:
            scores = test(d)
            scores_dict_table(scores, f'RandomForest_{test.__name__}_{d.name}')

    for test in [test_n_estimators]:
        for d in dss:
            scores = test(d)
            flatplot(scores, f'RandomForest_{test.__name__}_{d.name}')
