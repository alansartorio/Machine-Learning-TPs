import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from typing import Callable, Dict, List, Tuple
import math
import random
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'

sns.set_theme()


# split_fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
split_fracs = [0.8]
for split_frac in split_fracs:
    data = pd.read_csv(f'data/split_{split_frac:02}.csv', index_col='index')
    data.rename(columns={
        "expected": "Real",
        "prediction": "Predicción"
    }, inplace=True, errors='raise')
    groups = data.groupby(['Real', 'Predicción']).count()
    # Pivot
    groups = groups.reset_index().pivot(index='Real', columns=['Predicción'])['probability'].fillna(0)
    # Normalize rows
    groups = groups.div(groups.sum(axis=1), axis=0)
    print(groups)
    plt.figure(figsize=(8, 7))
    # plt.title("Matriz de confusión")
    sns.heatmap(groups, annot=True, cmap="Blues", fmt="0.2f", vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig("plots/2_confusion_matrix.svg")
    # plt.show()
    plt.clf()

    # print(data)