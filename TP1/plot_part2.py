import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from typing import Callable, Dict, List, Tuple, Set, Optional
import math
import random
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool

show_plots = os.environ.get("SHOW_PLOTS", "1") == "1"

sns.set_theme()

points = DataFrame(columns=["TFP", "TVP", "Categoria", "Training %"])


def compute_metrics_per_class(
    data: DataFrame,
    classes: Set[str],
    threshold: float,
    tag_name: str,
    save_results=False,
) -> Tuple[float, float]:
    metrics = {"true_positive", "false_negative", "false_positive", "true_negative"}
    class_metrics = dict()
    for c in classes:
        initial_metrics = dict()
        for metric in metrics:
            initial_metrics[metric] = 0
        class_metrics[c] = initial_metrics

    for _, row in data.iterrows():
        for class_ in classes:
            is_class_prediction = row[f"probability_{class_}"] > threshold
            is_class_real: bool = row[tag_name] == class_

            if is_class_real and is_class_prediction:
                result = "true_positive"
            elif is_class_real and not is_class_prediction:
                result = "false_negative"
            elif not is_class_real and is_class_prediction:
                result = "false_positive"
            else:
                result = "true_negative"
            class_metrics[class_][result] += 1

    for class_ in classes:
        true_positive = class_metrics[class_]["true_positive"]
        true_negative = class_metrics[class_]["true_negative"]
        false_positive = class_metrics[class_]["false_positive"]
        false_negative = class_metrics[class_]["false_negative"]

        def safe_divide(a, b) -> float:
            if b == 0:
                return 0
            return a / b

        accuracy = safe_divide(
            true_positive + true_negative,
            true_positive + true_negative + false_positive + false_negative,
        )
        precision = safe_divide(true_positive, (true_positive + false_positive))
        recall = true_positive_rate = safe_divide(
            true_positive, (true_positive + false_negative)
        )
        false_positive_rate = safe_divide(
            false_positive, (false_positive + true_negative)
        )
        f1_score = safe_divide(2 * precision * recall, (precision + recall))

        class_metrics[class_]["accuracy"] = accuracy
        class_metrics[class_]["precision"] = precision
        class_metrics[class_]["false_positive_rate"] = false_positive_rate
        class_metrics[class_]["true_positive_rate"] = true_positive_rate
        class_metrics[class_]["f1_score"] = f1_score

    if save_results:
        with open("out/part2/metrics.json", "+w") as f:
            json.dump(class_metrics, f, indent=4)

    return class_metrics


def plot_roc(data: DataFrame, show_fig=True, save_path: Optional[str] = None) -> None:
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 8))

    for class_name in df_roc["Categoria"].unique():
        class_data = df_roc[df_roc["Categoria"] == class_name]
        ax = sns.lineplot(
            data=class_data,
            x="TFP",
            y="TVP",
            label=class_name,
            marker="o",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Clas. aleatorio")
    plt.xlabel("TFP")
    plt.ylabel("TVP")
    plt.title("Curva ROC por categoría")
    plt.legend(title="Categorías")
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    plt.clf()


def plot_metrics(
    data: DataFrame, show_fig=True, save_path: Optional[str] = None
) -> None:
    highest_frac = data.melt(
        id_vars=["Categoria"],
        value_vars=["TFP", "TVP", "Exactitud", "Precisión", "F1-score"],
        var_name="Metrica",
        value_name="Valor",
    )

    plt.figure(figsize=(14, 9))
    sns.barplot(
        data=highest_frac, y="Valor", x="Categoria", hue="Metrica", errorbar=None
    )
    plt.title("Métricas")
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    plt.clf()


def plot_confusion_matrix(
    data: DataFrame, show_fig=False, save_path: Optional[str] = None
) -> None:
    data.rename(
        columns={"expected": "Real", "prediction": "Predicción"},
        inplace=True,
        errors="raise",
    )
    groups = data.groupby(["Real", "Predicción"]).count()
    # Pivot
    groups = (
        groups.reset_index()
        .pivot(index="Real", columns=["Predicción"])["probability_prediction"]
        .fillna(0)
    )

    # Normalize rows
    groups = groups.div(groups.sum(axis=1), axis=0).mul(100)
    plt.figure(figsize=(8, 7))
    plt.title("Matriz de confusión")
    sns.heatmap(groups, annot=True, cmap="Blues", fmt="0.1f", vmin=0, vmax=100)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    plt.clf()


split_fracs = [0.7]
# split_fracs = [0.8]
for split_frac in split_fracs:
    data = pd.read_csv(f"data/split_{split_frac:02}.csv", index_col="index")

    classes = data["expected"].unique()

    roc_data = []
    for threshold in np.arange(0, 0.9, 0.1):
        metrics = compute_metrics_per_class(data, classes, threshold, "prediction")
        # points.append(metrics)
        for class_name, metrics in metrics.items():
            roc_data.append(
                {
                    "Categoria": class_name,
                    "TFP": metrics["false_positive_rate"],
                    "TVP": metrics["true_positive_rate"],
                    "Umbral": threshold,
                    "Precisión": metrics["precision"],
                    "Exactitud": metrics["accuracy"],
                    "F1-score": metrics["f1_score"],
                }
            )

    # with open('out/part2/metrics.json', '+w') as f:
    #         json.dump(points, f, indent=4)

    df_roc = pd.DataFrame(roc_data)

    plot_roc(df_roc, show_fig=False, save_path="plots/2_roc_curve.svg")

    best_threshold = 0.3
    df_metrics = df_roc.drop(df_roc[df_roc["Umbral"] == best_threshold].index)

    plot_metrics(df_metrics, show_fig=False, save_path="plots/2_metrics.svg")

    plot_confusion_matrix(
        data, show_fig=False, save_path="plots/2_confusion_matrix.svg"
    )

    exit()
    data.rename(
        columns={"expected": "Real", "prediction": "Predicción"},
        inplace=True,
        errors="raise",
    )
    groups = data.groupby(["Real", "Predicción"]).count()
    # Pivot
    groups = (
        groups.reset_index()
        .pivot(index="Real", columns=["Predicción"])["probability"]
        .fillna(0)
    )

    for clazz in groups.columns:
        print(clazz)
        row = groups[groups.index == clazz]
        col = groups[clazz]
        true_positive = float(row[clazz].iloc[0])
        false_negative = float(row.loc[:, row.columns != clazz].sum(axis=1).iloc[0])
        false_positive = col[col.index != clazz].sum(axis=0)

        other_rows = groups[groups.index != clazz]
        true_negative = (
            other_rows.loc[:, other_rows.columns != clazz].sum(axis=1).sum(axis=0)
        )

        data = [
            ["Positivo", "Positiva", true_positive],
            ["Positivo", "Negativa", false_negative],
            ["Negativo", "Positiva", false_positive],
            ["Negativo", "Negativa", true_negative],
        ]

        matrix = DataFrame(data, columns=["Real", "Predicción", "Cantidad"])
        matrix = matrix.pivot(index="Real", columns=["Predicción"])["Cantidad"]
        matrix.sort_index(axis=0, ascending=False, inplace=True)
        matrix.sort_index(axis=1, ascending=False, inplace=True)
        print(matrix)
        # sns.heatmap(matrix, annot=True, cmap="Blues", fmt=".0f", vmin=0)
        # plt.show()
        print(true_positive)
        print(false_positive)
        print(false_negative)
        print(true_negative)

        accuracy = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )
        presicion = true_positive / (true_positive + false_positive)
        recall = true_positive_rate = true_positive / (true_positive + false_negative)
        false_positive_rate = false_positive / (false_positive + true_negative)
        f1_score = (2 * presicion * recall) / (presicion + recall)

        print("Accuracy", accuracy)
        print("Presicion", presicion)
        print("Recall", recall)
        print("False Positive Rate", false_positive_rate)
        print("F1-score", f1_score)

        points = pd.concat(
            [
                points,
                DataFrame(
                    {
                        "TFP": [false_positive_rate],
                        "TVP": [true_positive_rate],
                        "Accuracy": [accuracy],
                        "Precisión": [presicion],
                        "F1-score": [f1_score],
                        "Categoria": [clazz],
                        "Training %": [split_frac],
                    }
                ),
            ]
        )

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
print(points["Training %"].map(str))
sns.lineplot(
    data=points,
    x="TFP",
    y="TVP",
    hue="Categoria",
    # linewidth=points['Training %'] * 100,
    # sizes=(.2, .8),
    # style=points['Training %'].map(str),
    marker="o",
)
plt.tight_layout()
plt.savefig("plots/2_roc_curve.svg")
# plt.show()
plt.clf()


highest_frac = points[points["Training %"] == split_fracs[-1]]
highest_frac = highest_frac.melt(
    id_vars=["Categoria"],
    value_vars=["TFP", "TVP", "Accuracy", "Precisión", "F1-score"],
    var_name="Metrica",
    value_name="Valor",
)
print(highest_frac)

plt.figure(figsize=(14, 9))
sns.barplot(data=highest_frac, y="Valor", x="Categoria", hue="Metrica")
plt.tight_layout()
plt.savefig("plots/2_metrics.svg")

# plt.show()
