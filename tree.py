from typing import List

from sklearn.tree import DecisionTreeClassifier

from const import default_c
from datasets import Data
from experiments import test_param
from plotter import scores_dict_table


def test_criterion(data: Data):
    return test_param(DecisionTreeClassifier, data, default_c, 'criterion', ('gini', 'entropy'))


def test_splitter(data: Data):
    return test_param(DecisionTreeClassifier, data, default_c, 'splitter', ('best', 'random'))


def test_max_depth(data: Data):
    return test_param(DecisionTreeClassifier, data, default_c, 'max_depth', (1, 3, 8, 13, 21, None))


def test_min_samples_split(data: Data):
    return test_param(DecisionTreeClassifier, data, default_c, 'min_samples_split', (2, 3, 8, 13, 21, 34))


def run(dss: List[Data]):
    for d in dss:
        for test in [test_criterion, test_splitter, test_max_depth, test_min_samples_split]:
            scores = test(d)
            scores_dict_table(scores, f'DecisionTree_{test.__name__}_{d.name}')
