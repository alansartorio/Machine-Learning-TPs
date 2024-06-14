from itertools import count
import numpy as np
import numpy.typing as npt
from functools import partial
import polars as pl
from dataset import (
    load_dataset,
    budget,
    production_companies,
    production_countries,
    popularity,
    revenue,
    runtime,
    spoken_languages,
    vote_average,
    vote_count,
)


def sq_distance(
    point: npt.NDArray[np.float64], centroid: npt.NDArray[np.float64]
) -> np.float64:
    # print(centroid, point)
    return np.linalg.norm(centroid - point, ord=2)


def wcss(
    points: npt.NDArray[np.float64], centroid: npt.NDArray[np.float64]
) -> np.float64:
    point_dists = np.apply_along_axis(
        partial(sq_distance, centroid=centroid), axis=1, arr=points
    )
    # print(point_dists)
    return point_dists.sum()


def wcss_total(
    df: pl.DataFrame, centroids: pl.DataFrame, variables: list[str]
) -> float:
    return (
        centroids.select(variables)
        .map_rows(
            lambda centroid: wcss(df.select(variables).to_numpy(), np.array(centroid))
        )
        .to_numpy()
        .sum()
    )


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def plot_clusters(
    df: pl.DataFrame,
    cluster_indices: npt.NDArray[np.int64],
    variables_to_show: list[str],
    centroids: pl.DataFrame,
):
    df = df.select(variables_to_show)
    sns.scatterplot(
        data=df, x=variables_to_show[0], y=variables_to_show[1], hue=cluster_indices
    )
    sns.scatterplot(
        data=centroids,
        x=variables_to_show[0],
        y=variables_to_show[1],
        hue=range(len(centroids)),
        s=100,
    )
    plt.show()


def k_means_inner(
    df: pl.DataFrame, centroids: pl.DataFrame, variables: list[str]
) -> pl.DataFrame:
    points = df.select(variables)
    closest_centroid = lambda point: np.argmin(
        np.apply_along_axis(
            lambda centroid: sq_distance(point, centroid),
            axis=1,
            arr=centroids.select(variables).to_numpy(),
        )
    )
    centroid_indices = np.apply_along_axis(closest_centroid, axis=1, arr=points)
    # plot_clusters(df, centroid_indices, [budget, revenue], centroids)

    new_centroids = (
        df.select(variables)
        .with_columns(pl.Series(centroid_indices).alias("centroid"))
        .group_by("centroid")
        .mean()
        .sort("centroid")
        .drop("centroid")
    )

    return new_centroids


# TODO: DONT DROP NULLS
df = load_dataset().drop_nulls()
numeric_vars = [
    budget,
    popularity,
    production_companies,
    production_countries,
    revenue,
    runtime,
    spoken_languages,
    vote_average,
    vote_count,
]

mins = df.select(numeric_vars).min().to_numpy()
maxs = df.select(numeric_vars).max().to_numpy()
spread = maxs - mins


def run_k_means(centroids: pl.DataFrame):
    iterations = {"iteration": [], "error": []}
    last = None
    iterations_since_last_improvement = 0

    for i in count(1):
        error = wcss_total(df, centroids, numeric_vars)
        print("ITERATION: ", i, " | WCSS: ", error)
        centroids = k_means_inner(df, centroids, numeric_vars)
        iterations["iteration"].append(i)
        iterations["error"].append(error)
        if last is None or error < last:
            iterations_since_last_improvement = 0
        else:
            iterations_since_last_improvement += 1
        if iterations_since_last_improvement > 10:
            break
        last = error

    return pl.DataFrame(iterations)


results = []

try:
    for k in range(1, 10):
        for run in range(12):
            centroids = pl.from_numpy(
                np.random.rand(k, len(numeric_vars)) * spread + mins,
                schema=numeric_vars,
            )

            results.append(
                run_k_means(centroids).with_columns(
                    pl.lit(k).alias("k"), pl.lit(run).alias("run")
                )
            )
except KeyboardInterrupt:
    pass

pl.concat(results).write_csv("out/iterations.csv")
