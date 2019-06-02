from typing import List

from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from CONST import default_c
from datasets import Data
from experiments import test_param
from plotter import scores_dict_table, flatplot


def test_n_estimators(data: Data, base_estimator):
    c_params = {
        'random_state': 1,
        'base_estimator': base_estimator
    }
    return test_param(BaggingClassifier, data, c_params, 'n_estimators', [*range(1, 26)], s=('f1_micro',))


def test_max_samples(data: Data, base_estimator):
    c_params = {
        'base_estimator': base_estimator,
        **default_c
    }
    return test_param(BaggingClassifier, data, c_params, 'max_samples', (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0))


def test_max_features(data: Data, base_estimator):
    c_params = {
        'base_estimator': base_estimator,
        **default_c
    }
    return test_param(BaggingClassifier, data, c_params, 'max_features', (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0))


def test_bootstrap(data: Data, base_estimator):
    c_params = {
        'base_estimator': base_estimator,
        **default_c
    }
    return test_param(BaggingClassifier, data, c_params, 'bootstrap', (True, False))


def run(dss: List[Data]):
    estimators, params = (DecisionTreeClassifier, GaussianNB), ({'random_state': 1}, {})
    for d in dss:
        for estim, p in zip(estimators, params):
            for test in [test_max_samples, test_max_features, test_bootstrap]:
                scores = test(d, estim(**p))
                scores_dict_table(scores, f'Bagging_{estim.__name__}_{test.__name__}_{d.name}')

            for test in [test_n_estimators]:
                scores = test(d, estim(**p))
                flatplot(scores, f'Bagging_{estim.__name__}_{test.__name__}_{d.name}')
