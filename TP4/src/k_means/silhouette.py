if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from functools import partial
from itertools import islice
import polars as pl
from tqdm import tqdm
import dataset
from dataset import DatasetType, load_dataset
from k_means import classify
from k_means import FloatArray, sq_distance
import k_means
import numpy as np


INPUT = "out/k_means/centroids_all_columns.csv"
OUTPUT = "out/k_means/silhouette.csv"


def silhouette(df: pl.DataFrame, centroids: pl.DataFrame):
    def avg_dst(point: FloatArray, points: FloatArray):
        return np.linalg.norm(point - points, axis=1).mean()

    def avg_dst_with_nth_cluster(point, n: int):
        nth_cluster = (
            centroids.lazy()
            .with_row_index("cluster")
            .sort(
                pl.concat_list(pl.all().exclude("cluster")).map_elements(
                    lambda centroid: sq_distance(np.array(point), centroid),
                    return_dtype=pl.Float64,
                )
            )
            .filter(pl.col("cluster").is_in(points_by_cluster.keys()))
            .limit(n + 1)
            .last()
            .collect()
            .row(0, named=True)["cluster"]
        )
        return avg_dst(point, points_by_cluster[nth_cluster])

    def avg_dst_within_cluster(point: FloatArray):
        return avg_dst_with_nth_cluster(point, 0)

    def avg_dst_nearest_cluster(point: FloatArray):
        return avg_dst_with_nth_cluster(point, 1)

    def silhouette_coefficient(point: FloatArray):
        cohesion = avg_dst_within_cluster(point)
        separation = avg_dst_nearest_cluster(point)

        return (separation - cohesion) / max(separation, cohesion)

    clusters = classify(df, centroids)
    categorized = df.with_columns(pl.Series(clusters).alias("cluster"))
    # print(categorized.get_column("cluster").value_counts().sort("cluster"))
    points_by_cluster = {
        cluster: points.drop("cluster").to_numpy()
        for (cluster,), points in categorized.group_by(("cluster",))
    }
    # print(points_by_cluster.keys())

    with tqdm(total=len(df), disable=True) as pbar:
        return (
            df.select(
                coef=pl.concat_list(pl.all()).map_elements(
                    k_means.w_pbar(
                        pbar, lambda point: silhouette_coefficient(np.array(point))
                    ),
                    return_dtype=pl.Float64,
                )
            )
            .get_column("coef")
            .mean()
        )


def group_to_silhouette(df, args):
    group, centroids = args
    return (*group, silhouette(df, centroids.drop(("k", "run"))))


# centroids_all_runs.group_by('k', 'run').map_groups(lambda centroids: centroids.group_by)
# exit(1)

if __name__ == "__main__":
    from multiprocessing import get_context
    import csv

    df = load_dataset(DatasetType.NORMALIZED)
    df = df.select(k_means.all_numeric_columns)

    centroids_all_runs = pl.read_csv(INPUT)

    with get_context("spawn").Pool(12) as p, open(OUTPUT, "w") as output_file:
        writer = csv.writer(output_file)
        total = centroids_all_runs.n_unique(("k", "run"))
        writer.writerow(("k", "run", "silhouette"))
        writer.writerows(
            tqdm(
                p.imap_unordered(
                    partial(group_to_silhouette, df),
                    centroids_all_runs.group_by("k", "run"),
                ),
                total=total,
            )
        )

    # with tqdm(total=centroids_all_runs.n_unique(("k", "run"))) as pbar:
    # centroids_all_runs.group_by("k", "run").map_groups(
    # k_means.w_pbar(
    # pbar, lambda centroids: silhouette(centroids.drop(("k", "run")))
    # ),
    # ).write_csv(OUTPUT)
