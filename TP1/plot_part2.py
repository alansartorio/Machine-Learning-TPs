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

points = DataFrame(columns=['TFP', 'TVP', 'Categoria', 'Training %'])

split_fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# split_fracs = [0.8]
for split_frac in split_fracs:
    data = pd.read_csv(f'data/split_{split_frac:02}.csv', index_col='index')
    data.rename(columns={
        "expected": "Real",
        "prediction": "Predicción"
    }, inplace=True, errors='raise')
    groups = data.groupby(['Real', 'Predicción']).count()
    # Pivot
    groups = groups.reset_index().pivot(index='Real', columns=['Predicción'])['probability'].fillna(0)
    
    for clazz in groups.columns:
        print(clazz)
        row = groups[groups.index == clazz]
        col = groups[clazz]
        true_positive = float(row[clazz].iloc[0])
        false_negative = float(row.loc[:, row.columns != clazz].sum(axis=1).iloc[0])
        false_positive = col[col.index != clazz].sum(axis=0)

        other_rows = groups[groups.index != clazz]
        true_negative = other_rows.loc[:, other_rows.columns != clazz].sum(axis=1).sum(axis=0)

        data = [
            ['Positivo', 'Positiva', true_positive],
            ['Positivo', 'Negativa', false_negative],
            ['Negativo', 'Positiva', false_positive],
            ['Negativo', 'Negativa', true_negative],
        ]

        matrix = DataFrame(data, columns=['Real', 'Predicción', 'Cantidad'])
        matrix = matrix.pivot(index='Real', columns=['Predicción'])['Cantidad']
        matrix.sort_index(axis=0, ascending=False, inplace=True)
        matrix.sort_index(axis=1, ascending=False, inplace=True)
        print(matrix)
        # sns.heatmap(matrix, annot=True, cmap="Blues", fmt=".0f", vmin=0)
        # plt.show()
        print(true_positive)
        print(false_positive)
        print(false_negative)
        print(true_negative)

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        presicion = true_positive / (true_positive + false_positive)
        recall = true_positive_rate = true_positive / (true_positive + false_negative)
        false_positive_rate = false_positive / (false_positive + true_negative)
        f1_score = (2 * presicion * recall) / (presicion + recall)

        print("Accuracy", accuracy)
        print("Presicion", presicion)
        print("Recall", recall)
        print("False Positive Rate", false_positive_rate)
        print("F1-score", f1_score)

        points = pd.concat([points, DataFrame({
            "TFP": [false_positive_rate],
            "TVP": [true_positive_rate],
            "Accuracy": [accuracy],
            "Precisión": [presicion],
            "F1-score": [f1_score],
            "Categoria": [clazz],
            "Training %": [split_frac]
        })])

    if split_frac == split_fracs[-1]:
        # Normalize rows
        groups = groups.div(groups.sum(axis=1), axis=0).mul(100)
        plt.figure(figsize=(8, 7))
        # plt.title("Matriz de confusión")
        sns.heatmap(groups, annot=True, cmap="Blues", fmt="0.1f", vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig("plots/2_confusion_matrix.svg")
        # plt.show()
        plt.clf()


print(points)
print(points['Training %'].map(str))
sns.lineplot(
        data=points,
        x='TFP',
        y='TVP',
        hue='Categoria',
        # linewidth=points['Training %'] * 100,
        # sizes=(.2, .8),
        # style=points['Training %'].map(str),
        marker='o',
)
plt.tight_layout()
plt.savefig('plots/2_roc_curve.svg')
# plt.show()
plt.clf()


highest_frac = points[points['Training %'] == split_fracs[-1]]
highest_frac = highest_frac.melt(
        id_vars=['Categoria'],
        value_vars=['TFP','TVP','Accuracy','Precisión','F1-score'],
        var_name='Metrica',
        value_name='Valor',
)
print(highest_frac)

plt.figure(figsize=(14, 9))
sns.barplot(data=highest_frac, y='Valor', x='Categoria', hue='Metrica')
plt.tight_layout()
plt.savefig('plots/2_metrics.svg')

# plt.show()
