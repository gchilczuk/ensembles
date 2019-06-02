import bagging
import boosting
import random_forest
import tree
from datasets import pima, glass, wine, iris

if __name__ == '__main__':
    tree.run([iris])
    random_forest.run([pima, glass, wine])
    bagging.run([pima, glass, wine])
    boosting.run([pima, glass, wine])
