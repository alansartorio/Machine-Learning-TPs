if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from typing import Iterable, Optional
import polars as pl
from tqdm import tqdm
from k_means import classify
import k_means
import numpy as np
from k_means.k_means import FloatArray
import scipy.cluster.hierarchy as h
import time
import os


# np.set_printoptions(threshold=sys.maxsize)
def classify_in_column(
    df_normalized: pl.DataFrame,
    df_numerical: pl.DataFrame,
    t: float,
    linkage: FloatArray,
) -> tuple[int, pl.DataFrame]:
    # clusters = h.fcluster(linkage, k, criterion="maxclust") - 1
    clusters = h.fcluster(linkage, t, criterion="distance") - 1
    clusters = pl.Series(clusters)
    amount_of_clusters = clusters.n_unique()
    print("amount of clusters:", amount_of_clusters)
    print("observations in each cluster:", clusters.value_counts())

    categorized = df_numerical.with_columns(clusters.alias("cluster"))
    return amount_of_clusters, categorized


def pivot(
    categorized: pl.DataFrame,
    cluster_order: Optional[Iterable[int]] = None,
    normalize=False,
    index: Optional[str] = None,
    column: Optional[str] = None,
):
    if index is None:
        index = dataset.genres
    if column is None:
        column = "cluster"
    categorized = categorized.pivot(
        index=index,
        columns=column,
        values=dataset.imdb_id,
        aggregate_function=pl.len(),
        sort_columns=True,
    ).fill_null(0)

    def scale_to_sum_1(values):
        values = np.array(values)
        return tuple(values / values.sum() * 100)

    if normalize:
        columns = [col for col in categorized.columns if col != index]
        categorized = categorized.select(
            index,
            values=pl.concat_list(pl.all().exclude(index))
            .map_elements(scale_to_sum_1, return_dtype=pl.List(pl.Float64))
            .list.to_struct(fields=columns.__getitem__),
        ).unnest("values")

    if cluster_order is not None:
        categorized = categorized.select(index, *map(str, cluster_order))
    return categorized


def plot_confusion(
    categorized: pl.DataFrame,
    cluster_order: Optional[Iterable[int]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[str] = None,
    actual: Optional[str] = None,
    predicted: Optional[str] = None,
):
    categorized = pivot(
        categorized, cluster_order, normalize=True, index=actual, column=predicted
    )

    ax = sns.heatmap(
        categorized.to_pandas().set_index(actual),
        annot=True,
        cmap="Blues",
        fmt="0.1f",
        vmin=0,
        vmax=100,
        yticklabels=True,
    )
    for t in ax.texts:
        t.set_text(t.get_text() + "%")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if 'HIDE_PLOTS' not in os.environ:
        plt.show()


def plot_classification(
    categorized: pl.DataFrame, cluster_order: Optional[Iterable[int]] = None
):
    plot_confusion(
        categorized,
        cluster_order,
        xlabel="Cluster",
        ylabel="Genre",
        filename=OUTPUT_CLUSTERS,
        actual=dataset.genres,
        predicted="cluster",
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import dataset
    from dataset import DatasetType, load_dataset

    sns.set_theme()

    OUTPUT_CLUSTERS = "plots/hierarchical/classification.svg"
    OUTPUT_CONFUSION = "plots/hierarchical/confusion.svg"

    import argparse
    from __init__ import all_numeric_columns, just_numeric_columns

    parser = argparse.ArgumentParser(prog="k_means")
    parser.add_argument("dataset", choices=["numeric-columns", "all-columns"])

    args = parser.parse_args()

    match args.dataset:
        case "all-columns":
            # TODO: should we add dataset.release_date?
            numeric_vars = all_numeric_columns
            output_file_variant = "all_columns"
        case "numeric-columns":
            numeric_vars = just_numeric_columns
            output_file_variant = "numeric_columns"
        case dataset:
            raise Exception(f"Invalid dataset {dataset}")
    INPUT = f"out/hierarchical/linkage_{output_file_variant}.csv"
    OUTPUT = f"plots/hierarchical/dendogram_{output_file_variant}.svg"

    linkage = np.loadtxt(INPUT)
    cut = len(linkage) + 1

    k = 200

    def filter_genres(df):
        return df.filter(pl.col(dataset.genres).is_in(("Action", "Comedy", "Drama")))

    df_normalized = filter_genres(load_dataset(DatasetType.NORMALIZED)).sample(cut, seed=1346789134)
    df_numerical = filter_genres(load_dataset(DatasetType.NUMERICAL)).sample(cut, seed=1346789134)

    def assign_labels_to_clusters(t: float) -> tuple[int, pl.DataFrame, dict[int, str]]:
        k, categorized = classify_in_column(df_normalized, df_numerical, t, linkage)

        pivoted = pivot(categorized, normalize=True)

        assignation = {
            cluster: pivoted[dataset.genres][pivoted[cluster].arg_max()]
            for cluster in pivoted.columns[1:]
        }
        # print(pivoted)
        # print(assignation)

        categorized = categorized.with_columns(
            predicted=pl.col("cluster").replace(assignation, default=None)
        )

        # Example output
        return k, categorized, assignation

    def score_assignation(categorized: pl.DataFrame) -> np.float64:
        counts = (
            categorized.select(dataset.genres, "predicted", dataset.imdb_id)
            .pivot(
                index=dataset.genres,
                columns="predicted",
                values=dataset.imdb_id,
                aggregate_function=pl.len(),
                sort_columns=True,
            )
            .fill_null(0)
        )

        # print(counts)
        raise NotImplementedError()

    # Multiclass accuracy formula
    # https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    def accuracy_score(vals: pl.DataFrame) -> float:
        return len(vals.filter(pl.col("actual") == pl.col("predicted"))) / len(vals)

    a, b = 0, 1
    desired_k = 100
    k = None
    while k != desired_k:
        d = (a + b) / 2
        k, classified = classify_in_column(
            df_normalized, df_numerical, t=d, linkage=linkage
        )
        print(d, k)
        if k < desired_k:
            b = d
        else:
            a = d
    print(f"FOUND k={desired_k} for d={d}!")
    plot_classification(classified, range(k))

    k, categorized, assignations = assign_labels_to_clusters(d)
    print(assignations)

    plot_confusion(
        categorized.select(
            dataset.imdb_id, actual=dataset.genres, predicted="predicted"
        ),
        actual="actual",
        predicted="predicted",
        xlabel="predicted",
        ylabel="actual",
        filename=OUTPUT_CONFUSION,
    )
