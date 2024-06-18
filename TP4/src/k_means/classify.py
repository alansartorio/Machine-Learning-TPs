if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
from typing import Iterable, Optional
import polars as pl
from tqdm import tqdm
from k_means import classify
import k_means
import numpy as np


# np.set_printoptions(threshold=sys.maxsize)
def classify_in_column(
    df_normalized: pl.DataFrame, df_numerical: pl.DataFrame, centroids: pl.DataFrame
):
    clusters = classify(df_normalized.select(k_means.all_numeric_columns), centroids)

    categorized = df_numerical.with_columns(pl.Series(clusters).alias("cluster"))
    return categorized


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
        if set(categorized.columns) >= set(cluster_order):
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
    if "HIDE_PLOTS" not in os.environ:
        plt.show()
    plt.clf()


def plot_classification(
    categorized: pl.DataFrame, k: int, cluster_order: Optional[Iterable[int]] = None
):
    plot_confusion(
        categorized,
        cluster_order,
        xlabel="Cluster",
        ylabel="Genre",
        filename=OUTPUT_CLUSTERS % k,
        actual=dataset.genres,
        predicted="cluster",
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import dataset
    from dataset import DatasetType, load_dataset

    sns.set_theme()

    OUTPUT_CLUSTERS = "plots/k_means/classification_%s.svg"
    OUTPUT_CONFUSION = "plots/k_means/confusion_%s.svg"

    all_centroids = pl.read_csv("out/k_means/centroids_all_columns.csv")

    def classify_with_k(k: int):
        def filter_genres(df):
            return df.filter(
                pl.col(dataset.genres).is_in(("Action", "Comedy", "Drama"))
            )

        df_normalized = filter_genres(load_dataset(DatasetType.NORMALIZED))
        df_numerical = filter_genres(load_dataset(DatasetType.NUMERICAL))

        def assign_labels_to_clusters(
            run: int, k: int
        ) -> tuple[pl.DataFrame, dict[int, str]]:
            centroids = all_centroids.filter(
                (pl.col("k") == k) & (pl.col("run") == run)
            ).drop("k", "run")

            categorized = classify_in_column(df_normalized, df_numerical, centroids)

            print(categorized["cluster"].n_unique())
            if categorized["cluster"].n_unique() != k:
                return None, None

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
            return categorized, assignation

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

        def assign_and_score_run(run: int):
            # centroid_count = len(all_centroids.filter((pl.col("k") == k) & (pl.col("run") == run)))
            # if centroid_count != k:
                # return 0
            categorized, assignation = assign_labels_to_clusters(run, k)
            if categorized is None: return 0
            return accuracy_score(
                categorized.select(actual=dataset.genres, predicted="predicted")
            )
            # return score_assignation(categorized)

        best_run = max(
            tqdm(all_centroids.filter(pl.col("k") == k).get_column("run").unique()),
            key=assign_and_score_run,
        )

        best_centroids = all_centroids.filter(
            (pl.col("k") == k) & (pl.col("run") == best_run)
        ).drop("k", "run")

        plot_classification(
            classify_in_column(df_normalized, df_numerical, best_centroids), k, range(k)
        )

        categorized, assignations = assign_labels_to_clusters(best_run, k)
        assert categorized is not None
        if categorized is None: raise Exception()
        print(assignations)

        plot_confusion(
            categorized.select(
                dataset.imdb_id, actual=dataset.genres, predicted="predicted"
            ),
            actual="actual",
            predicted="predicted",
            xlabel="predicted",
            ylabel="actual",
            filename=OUTPUT_CONFUSION % k,
        )

    classify_with_k(3)
    classify_with_k(5)
    classify_with_k(11)
