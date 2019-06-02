import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import preprocessing

_here = os.path.dirname(os.path.abspath(__file__))

_iris_file = os.path.join(_here, 'iris', 'iris.csv')
_pima_file = os.path.join(_here, 'pima-diabetes', 'diabetes.csv')
_glass_file = os.path.join(_here, 'glass', 'glass.csv')
_wine_file = os.path.join(_here, 'wine', 'wine.csv')

iris = pd.read_csv(_iris_file)
iris.name = 'iris'

pima = pd.read_csv(_pima_file)
pima.name = 'pima'

wine = pd.read_csv(_wine_file)
wine = wine[wine.columns.tolist()[1:] + wine.columns.tolist()[:1]]
wine.name = 'wine'

glass = pd.read_csv(_glass_file)
glass.name = 'glass'


@dataclass
class Data:
    x: np.ndarray
    y: np.ndarray
    name: str

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls(
            x=preprocessing.scale(df[df.columns[:-1]].values),
            # x=ds[ds.columns[:-1]].values,
            y=df['class'].values,
            name=df.name
        )


(iris, pima, wine, glass) = [Data.from_df(ds) for ds in (iris, pima, wine, glass)]

__all__ = [iris, pima, wine, glass]
