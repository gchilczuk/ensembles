import os

import numpy as np
import pandas as pd
import six
from matplotlib import pyplot as plt

from CONST import SHOW_IN_PLACE, IMG_DIR


def scores_dict_table(scores, filename):
    df = pd.DataFrame.from_dict(scores).round(3)
    render_mpl_table(df)
    plt.title(filename)
    if SHOW_IN_PLACE:
        plt.show()
    else:
        plt.savefig(os.path.join(IMG_DIR, filename))


def flatplot(scores, filename):
    df = pd.DataFrame.from_dict(scores).round(3)
    df.plot(marker='o')
    plt.title(filename)
    mi, ma = df.index.min(), df.index.max()
    plt.xticks(np.arange(mi - 1, ma + 1, (ma - mi) // 15))
    plt.xlabel('n_estimators')
    plt.ylabel('score')
    if SHOW_IN_PLACE:
        plt.show()
    else:
        plt.savefig(os.path.join(IMG_DIR, filename))


def render_mpl_table(data, col_width=2.0, row_height=0.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=1,
                     ax=None, **kwargs):
    data[''] = data.index
    cols = data.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax
