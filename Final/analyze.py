import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


@dataclass
class PlotConfig:
    show: bool = True
    output_path: Path = Path("plots")
    save_file: Optional[str] = None
    tight_layout: bool = False
    fig_size: Optional[Tuple[int, int]] = None

    def print_plot(self) -> None:
        if self.tight_layout:
            plt.tight_layout()
        if self.save_file is not None:
            plt.savefig(self.output_path.joinpath(self.save_file))
        if self.show:
            plt.show()
        plt.clf()


def plot_class_count(df: pd.DataFrame, config=PlotConfig()):
    print("Plotting class count...")
    cnt_pro = df["target"].value_counts()
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(x=cnt_pro.index, y=cnt_pro.values, palette=["red", "green"])
    plt.ylabel("Cantidad de diagnósticos", fontsize=12)
    plt.xlabel("target", fontsize=12)

    # Set x-tick labels
    barplot.set_xticklabels([" 0 = maligno ", " 1 =benigno"])

    config.print_plot()


# plt.rcParams.update({'font.size': 20})


def plot_corr_matrix(
    df: pd.DataFrame, cols: Optional[List[str]] = None, config=PlotConfig()
):
    plt.figure(figsize=(10, 9))
    h = sns.heatmap((df if cols is None else df[cols]).corr(), annot=True, linewidths=1)
    h.set_xticklabels(h.get_xticklabels(), rotation=45)
    h.set_yticklabels(h.get_yticklabels(), rotation=0)

    config.print_plot()


def plot_corr_target(
    df: pd.DataFrame,
    target="target",
    cols: Optional[List[str]] = None,
    config=PlotConfig(),
):
    fig, ax = plt.subplots(figsize=(6, 10))

    # Compute correlations and sort by absolute values
    corr = (
        (df if cols is None else df[cols])
        .corrwith(df[target])
        .sort_values(key=abs, ascending=False)
        .to_frame()
    )
    corr.columns = ["target"]

    sns.heatmap(corr, annot=True, linewidths=0.5, linecolor="black")
    plt.title("Matriz de correlación de las variables con la clasificación")

    config.print_plot()


def plot_distributions(
    df: pd.DataFrame, cols: Optional[List[str]] = None, config=PlotConfig()
):
    # Define the colors
    colors = ["red", "green"]
    ncols = 3
    nrows = len(cols) // ncols + 1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 30))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    ax_flat = ax.flatten()

    for i, col in enumerate(cols):
        plt.subplot(nrows, ncols, i + 1)
        ax = sns.violinplot(x="target", y=col, data=df, palette=colors)
        title = col + " vs target"
        plt.title(title, fontsize=10)

    for ax in ax_flat[i + 1 :]:
        ax.set_visible(False)
    for ax in ax_flat:
        for xlabel_i in ax.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        for xlabel_i in ax.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
        for tick in ax.get_xticklines():
            tick.set_visible(False)
        for tick in ax.get_yticklines():
            tick.set_visible(False)

    config.print_plot()


def evaluate_model(X_train, y_train, X_test, y_test):
    clf1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=43)
    clf1.fit(X_train, y_train)

    print(
        "Accuracy of Decision Tree classifier on original training set: {:.2f}".format(
            clf1.score(X_train, y_train)
        )
    )
    print(
        "Accuracy of Decision Tree classifier on original test set: {:.2f}".format(
            clf1.score(X_test, y_test)
        )
    )

    y_pred = clf1.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cnf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    # plt.show()
    plt.savefig("plots/confusion_matrix.svg")

    print(classification_report(y_test, y_pred))


def plot_data(X_, y_, titles, filename="dimensionality_reduction"):
    # Define the colors
    colors = ["red" if label == 0 else "green" for label in y_]

    # Create figure
    plt.figure(figsize=(16, 16))

    # Loop over each subplot
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        scatter = plt.scatter(X_[i][:, 0], X_[i][:, 1], c=colors, edgecolor="k")
        plt.title(titles[i])
        plt.gca().set_facecolor("lightgray")
        red_patch = mpatches.Patch(color="red", label="malignant")
        green_patch = mpatches.Patch(color="green", label="benign")
        plt.legend(handles=[red_patch, green_patch])

    # plt.show()
    plt.savefig(f"plots/{filename}.svg")


def evaluate_and_plot_model(X_train, y_train, X_test, y_test):
    clf1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=43)
    clf1.fit(X_train, y_train)

    print(
        "Accuracy of Decision Tree classifier on original training set: {:.2f}".format(
            clf1.score(X_train, y_train)
        )
    )
    print(
        "Accuracy of Decision Tree classifier on original test set: {:.2f}".format(
            clf1.score(X_test, y_test)
        )
    )

    y_pred = clf1.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with seaborn
    plt.figure(figsize=(7, 5))
    sns.heatmap(cnf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    print(classification_report(y_test, y_pred))

    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Predict on the mesh grid
    Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.8)

    colors = ["red" if label == 0 else "green" for label in y_train]

    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, edgecolor="k")
    plt.gca().set_facecolor("lightgray")
    red_patch = mpatches.Patch(color="red", label="malignant")
    green_patch = mpatches.Patch(color="green", label="benign")
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
