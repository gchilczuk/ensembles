import os

import bagging
import boosting
import random_forest
import tree
from const import IMG_DIR, SHOW_IN_PLACE
from datasets import pima, glass, wine, iris

if __name__ == '__main__':
    if not SHOW_IN_PLACE and not os.path.isdir(IMG_DIR):
        os.mkdir(IMG_DIR)
    tree.run([iris])
    random_forest.run([pima, glass, wine])
    bagging.run([pima, glass, wine])
    boosting.run([pima, glass, wine])
