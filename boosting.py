from typing import List

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from const import default_c
from datasets import Data
from experiments import test_param, test_dectree_meta, test_meta_param
from plotter import scores_dict_table, flatplot


def test_n_estimators(data: Data, base_estimator):
    c_params = {
        'random_state': 1,
        'base_estimator': base_estimator
    }
    return test_param(AdaBoostClassifier, data, c_params, 'n_estimators', [*range(1, 26)], s=('f1_micro',))


def test_learning_rate(data: Data, base_estimator):
    c_params = {
        'base_estimator': base_estimator,
        **default_c
    }
    return test_param(AdaBoostClassifier, data, c_params, 'learning_rate', (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0))


def test_algorithm(data: Data, base_estimator):
    c_params = {
        'base_estimator': base_estimator,
        **default_c
    }
    return test_param(AdaBoostClassifier, data, c_params, 'algorithm', ('SAMME', 'SAMME.R'))


def test_tree_depth(data: Data):
    x = [DecisionTreeClassifier(max_depth=i) for i in range(1, 5)]
    return test_dectree_meta(
        AdaBoostClassifier, data, dict(default_c),
        'base_estimator', x,
        'n_estimators', [*range(1, 30)]
    )


def test_tradeoff(data: Data, base_estimator):
    c_params = {
        'base_estimator': base_estimator,
        **default_c
    }
    return test_meta_param(
        AdaBoostClassifier, data, c_params,
        'learning_rate', (0.05, 0.2, 0.4, 0.8, 1.0),
        'n_estimators', [*range(1, 26)]
    )


def run(dss: List[Data]):
    estimators, params = (DecisionTreeClassifier, GaussianNB), ({'random_state': 1}, {})

    for d in dss:
        scores = test_tree_depth(d)
        flatplot(scores, f'Boosting_DecisionTree_test_tree_depth_{d.name}')

        for estim, p in zip(estimators, params):
            for test in []:
                scores = test(d, estim(**p))
                scores_dict_table(scores, f'Boosting_{estim.__name__}_{test.__name__}_{d.name}')

            for test in [test_tradeoff]:
                scores = test(d, estim(**p))
                flatplot(scores, f'Boosting_{estim.__name__}_{test.__name__}_{d.name}')
