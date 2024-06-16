if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain, count
from typing import Any, Iterable, Optional
import numpy as np
from numpy.core.multiarray import array
import numpy.typing as npt
from functools import partial
import polars as pl


import dataset
from k_means.k_means import FloatArray

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
    import plotly.figure_factory as ff
    import scipy.cluster.hierarchy as h
    import matplotlib.pyplot as plt

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
    output_file = f"plots/hierarchical/dendogram_{output_file_variant}.svg"

    df_np = df.select(numeric_vars).to_numpy()  # [:20,:]
    # df_np = np.array([[0, 0], [0, 1], [0, 3]])
    count, vars = df_np.shape

    class ClusterDistanceMethod(Enum):
        MAXIMUM = 0
        MINIMUM = 1
        MEAN = 2
        CENTROID = 3

    def mirror_triangular_matrix(dst: FloatArray) -> FloatArray:
        assert dst.shape[0] == dst.shape[1]
        i_lower = np.tril_indices(dst.shape[0], -1)
        dst[i_lower] = dst.T[i_lower]
        return dst

    def distance_between_pointgroups(
        cluster_a: FloatArray, cluster_b: Optional[FloatArray] = None
    ):
        symmetric = False
        if cluster_b is None:
            symmetric = True
            cluster_b = cluster_a

        a_len = cluster_a.shape[0]
        b_len = cluster_b.shape[0]
        dst = np.full((a_len, b_len), np.inf)

        if symmetric:
            for y in range(a_len):
                dst[y, y + 1 :] = np.linalg.norm(
                    cluster_a[y + 1 :] - cluster_a[y], axis=1
                )
            dst = mirror_triangular_matrix(dst)
        else:
            for a in range(a_len):
                dst[a, :] = np.linalg.norm(cluster_a[a, :] - cluster_b, axis=1)
        return dst

    def cluster_distance_matrix(
        clusters: list[FloatArray], method: ClusterDistanceMethod
    ):
        if method == ClusterDistanceMethod.CENTROID:
            return distance_between_pointgroups(
                np.array([cluster.mean(axis=1) for cluster in clusters])
            )

        count = len(clusters)
        dst = np.full((count, count), np.inf)
        # with get_context('spawn').Pool() as p:
        for y in tqdm.tqdm(range(count)):
            for x in range(y + 1, count):
                cluster_distances = distance_between_pointgroups(
                    clusters[y], clusters[x]
                )
                match method:
                    case ClusterDistanceMethod.MINIMUM:
                        value = cluster_distances.min()
                    case ClusterDistanceMethod.MAXIMUM:
                        value = cluster_distances.max()
                    case ClusterDistanceMethod.MEAN:
                        value = cluster_distances.mean()
                dst[y, x] = value
        dst = mirror_triangular_matrix(dst)
        return dst

    def flatten_cluster(cluster: Any) -> Iterable[int]:
        if isinstance(cluster, Iterable):
            for subcluster in cluster:
                yield from flatten_cluster(subcluster)

        assert isinstance(cluster, int)
        yield cluster

    def cluster_to_values(cluster: Iterable[int]) -> FloatArray:
        indices = np.array(tuple(cluster))
        return df_np[indices, :]

    clusters: list[Any] = list(range(count))
    while len(clusters) > 1:
        print(len(clusters))
        # print(clusters)
        # print(list(map(list, map(flatten_cluster, clusters))))
        # print(list(map(cluster_to_values, map(flatten_cluster, clusters))))
        dst = cluster_distance_matrix(
            list(map(cluster_to_values, map(flatten_cluster, clusters))),
            ClusterDistanceMethod.MINIMUM,
        )
        print(dst)
        a, b = np.unravel_index(dst.argmin(), dst.shape)
        print(a, b)
        clusters[a] = (a, b)
        clusters.pop(b)

    # z = h.single(dst)

    import sys

    sys.setrecursionlimit(10000)
    # truncate_cluster_count = 10
    # with plt.rc_context({'lines.linewidth': 0.5}):
    # h.dendrogram(z, truncate_cluster_count, truncate_mode='lastp', no_labels=True)
    # plt.savefig(output_file)

    # dendo = ff.create_dendrogram(df_np)
    # dendo.write_image("plots/hierarchical/dendogram.svg")

    # print(df)
