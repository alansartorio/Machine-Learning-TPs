if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from abc import ABC, abstractmethod
from itertools import count
import numpy as np
import numpy.typing as npt
from functools import partial
import polars as pl


FloatArray = npt.NDArray[np.float64]


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def sq_distance(point: FloatArray, centroid: FloatArray) -> np.float64:
    # print(centroid, point)
    return np.linalg.norm(centroid - point, ord=2)


def wcss(points: FloatArray, centroid: FloatArray) -> np.float64:
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


def plot_clusters(
    df: pl.DataFrame,
    cluster_indices: npt.NDArray[np.int64],
    variables_to_show: list[str],
    centroids: pl.DataFrame,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme()

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


def closest_centroid(
    point: FloatArray,
    centroids: FloatArray,
) -> np.intp:
    return np.argmin(
        np.apply_along_axis(
            lambda centroid: sq_distance(point, centroid),
            axis=1,
            arr=centroids,
        )
    )


def classify(
    points: FloatArray | pl.DataFrame,
    centroids: FloatArray | pl.DataFrame,
):
    if isinstance(points, pl.DataFrame):
        points = points.to_numpy()
    if isinstance(centroids, pl.DataFrame):
        centroids = centroids.to_numpy()

    closest_centroid_to_point = lambda point: closest_centroid(point, centroids)
    return np.apply_along_axis(closest_centroid_to_point, axis=1, arr=points)


def k_means_inner(
    df: pl.DataFrame, centroids: FloatArray, variables: list[str]
) -> tuple[np.float64, FloatArray]:
    points = df.select(variables)

    centroid_indices = classify(points.to_numpy(), centroids)
    # plot_clusters(df, centroid_indices, [budget, revenue], centroids)

    grouped = (
        df.select(variables)
        .with_columns(pl.Series(centroid_indices).alias("centroid"))
        .group_by(["centroid"])
    )

    total_error: np.float64 = np.float64(0)
    for (centroid_index,), group in grouped:
        assert type(centroid_index) == int
        total_error += wcss(
            group.select(variables).to_numpy(),
            centroids[centroid_index, :],
        )

    new_centroids = grouped.mean().sort("centroid").drop("centroid").to_numpy()

    return total_error, new_centroids


from dataclasses import dataclass


@dataclass
class CentroidGenerator(ABC):
    count: int

    @abstractmethod
    def generate(self) -> FloatArray: ...


@dataclass
class UniformCentroidGenerator(CentroidGenerator):
    mins: FloatArray
    spread: FloatArray

    def generate(self) -> FloatArray:
        return np.random.rand(self.count, len(self.mins)) * self.spread + self.mins


@dataclass
class SamplerCentroidGenerator(CentroidGenerator):
    points: FloatArray

    def generate(self) -> FloatArray:
        return np.random.default_rng().choice(self.points, self.count, replace=False)


def run_k_means(
    df: pl.DataFrame, centroids: FloatArray | CentroidGenerator, variables: list[str]
):
    if isinstance(centroids, CentroidGenerator):
        centroids = centroids.generate()

    assert isinstance(centroids, np.ndarray)

    minimum_error_delta = 1e9

    iterations = {"iteration": [], "error": []}
    last = None
    iterations_since_last_improvement = 0

    for i in count(0):
        # print("ITERATION: ", i, " | WCSS: ", error)
        error, centroids = k_means_inner(df, centroids, variables)
        iterations["iteration"].append(i)
        iterations["error"].append(error)
        if last is None or error < last - minimum_error_delta:
            iterations_since_last_improvement = 0
        else:
            iterations_since_last_improvement += 1
        if iterations_since_last_improvement > 2:
            break
        last = error

    return pl.from_numpy(centroids, variables), pl.DataFrame(iterations)


def run_k_means_aggregate(
    run: int,
    centroids: CentroidGenerator,
    df: pl.DataFrame,
    variables: list[str],
):
    best_centroids, iterations = run_k_means(df, centroids, variables)
    k_col = pl.lit(centroids.count).alias("k")
    run_col = pl.lit(run).alias("run")
    return best_centroids.with_columns(k_col, run_col), iterations.with_columns(
        k_col, run_col
    )


import dataset

all_numeric_columns = (
    dataset.budget,
    dataset.original_title_len,
    dataset.overview_len,
    dataset.popularity,
    dataset.production_companies,
    dataset.production_countries,
    dataset.revenue,
    dataset.runtime,
    dataset.spoken_languages,
    dataset.vote_average,
    dataset.vote_count,
)

just_numeric_columns = (
    dataset.budget,
    dataset.popularity,
    dataset.production_companies,
    dataset.production_countries,
    dataset.revenue,
    dataset.runtime,
    dataset.spoken_languages,
    dataset.vote_average,
    dataset.vote_count,
)

if __name__ == "__main__":
    from multiprocessing import get_context
    import csv
    import tqdm

    import argparse
    from dataset import DatasetType, load_dataset

    parser = argparse.ArgumentParser(prog="k_means")
    parser.add_argument("dataset", choices=["numeric-columns", "all-columns"])

    args = parser.parse_args()

    df = load_dataset(DatasetType.NORMALIZED)
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
    output_iterations_file = f"out/k_means/iterations_{output_file_variant}.csv"
    output_centroids_file = f"out/k_means/centroids_{output_file_variant}.csv"

    mins = df.select(numeric_vars).min().to_numpy()
    maxs = df.select(numeric_vars).max().to_numpy()
    spread = maxs - mins

    runs = 100

    with open(output_iterations_file, "w") as iterations_file, open(
        output_centroids_file, "w"
    ) as centroids_file:
        iterations_output = csv.writer(iterations_file)
        centroids_output = csv.writer(centroids_file)
        iteration_rows = ("k", "run", "iteration", "error")
        iterations_output.writerow(iteration_rows)
        centroid_rows = ("k", "run", *numeric_vars)
        centroids_output.writerow(centroid_rows)

        try:
            with get_context("forkserver").Pool(12) as p:
                for k in tqdm.tqdm(tuple(range(1, 20))):
                    results = tqdm.tqdm(
                        p.imap_unordered(
                            partial(
                                run_k_means_aggregate,
                                centroids=SamplerCentroidGenerator(
                                    count=k, points=df.select(numeric_vars).to_numpy()
                                ),
                                # centroids=UniformCentroidGenerator(
                                # count=k, mins=mins, spread=spread
                                # ),
                                df=df,
                                variables=numeric_vars,
                            ),
                            range(runs),
                        ),
                        total=runs,
                    )

                    for centroids, iterations in results:
                        centroids_output.writerows(
                            tuple(row[col] for col in centroid_rows)
                            for row in centroids.iter_rows(named=True)
                        )
                        centroids_file.flush()

                        iterations_output.writerows(
                            tuple(row[col] for col in iteration_rows)
                            for row in iterations.iter_rows(named=True)
                        )
                        iterations_file.flush()
        except KeyboardInterrupt:
            pass
