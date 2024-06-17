if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import dataset
from enum import Enum
from k_means.k_means import FloatArray
import numpy as np
from typing import Any, Iterable, Optional
from itertools import count
from functools import partial

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
            dst[y, y + 1 :] = np.linalg.norm(cluster_a[y + 1 :] - cluster_a[y], axis=1)
        dst = mirror_triangular_matrix(dst)
    else:
        for a in range(a_len):
            dst[a, :] = np.linalg.norm(cluster_a[a, :] - cluster_b, axis=1)
    return dst


def calculate_row(
    y, method: ClusterDistanceMethod = None, clusters: list[FloatArray] = None
):
    count = len(clusters)
    row_values = []
    for x in range(y + 1, count):
        cluster_distances = distance_between_pointgroups(clusters[y], clusters[x])
        match method:
            case ClusterDistanceMethod.MINIMUM:
                value = cluster_distances.min()
            case ClusterDistanceMethod.MAXIMUM:
                value = cluster_distances.max()
            case ClusterDistanceMethod.MEAN:
                value = cluster_distances.mean()
        row_values.append(value)
    return y, row_values


if __name__ == "__main__":
    from multiprocessing import get_context
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
    output_file = f"out/hierarchical/linkage_{output_file_variant}.csv"

    df_np = df.select(numeric_vars).sample(1000, seed=1346789134).to_numpy()
    # df_np = np.array(
        # [
            # [0, 0],
            # [0, 1],
            # [1, 2],
            # [1, 2.9],
        # ]
    # )
    # df_np = np.array([[0, 0], [0, 1], [0, 3]])
    count, vars = df_np.shape

    def cluster_distance_matrix(
        clusters: list[FloatArray], method: ClusterDistanceMethod
    ):
        if method == ClusterDistanceMethod.CENTROID:
            return distance_between_pointgroups(
                np.array([cluster.mean(axis=0) for cluster in clusters])
            )

        count = len(clusters)
        dst = np.full((count, count), np.inf)
        with get_context("spawn").Pool() as p:
            for y, row_values in p.imap_unordered(
                partial(calculate_row, method=method, clusters=clusters),
                tqdm.tqdm(range(count)),
            ):
                dst[y, y + 1 :] = row_values
        dst = mirror_triangular_matrix(dst)
        return dst

    # print(
    # cluster_distance_matrix(
    # clusters=[
    # np.array([[0, 0], [1, 1]]),
    # np.array([[1, 2], [3, 3]]),
    # np.array([[-3, -3], [-1, -2]]),
    # ],
    # method=ClusterDistanceMethod.MINIMUM,
    # )
    # )
    # exit()

    Node = tuple["Node", "Node", float, int] | int

    def flatten_cluster(cluster: Node) -> Iterable[int]:
        if isinstance(cluster, tuple):
            a, b, dist, idx = cluster
            yield from flatten_cluster(a)
            yield from flatten_cluster(b)
            return

        assert isinstance(
            cluster, (int, np.int64)
        ), f"cluster='{cluster}' should be an int, but it's a {type(cluster)}"
        yield cluster

    def cluster_to_values(cluster: Iterable[int]) -> FloatArray:
        indices = np.array(tuple(cluster))
        return df_np[indices, :]

    clusters: list[Any] = list(range(count))

    def get_cluster_id(cluster: Node):
        if isinstance(cluster, tuple):
            a, b, dist, idx = cluster
            return idx
        return cluster

    lines = []
    new_idx = count
    for i in tqdm.tqdm(range(len(clusters) - 1)):
        # print(clusters)
        # print(list(map(list, map(flatten_cluster, clusters))))
        # print(list(map(cluster_to_values, map(flatten_cluster, clusters))))
        dst = cluster_distance_matrix(
            list(map(cluster_to_values, map(flatten_cluster, clusters))),
            ClusterDistanceMethod.MAXIMUM,
        )
        # print(dst)
        a, b = np.unravel_index(dst.argmin(), dst.shape)
        # print(a, b)
        new_cluster = (clusters[a], clusters[b], dst[a, b], new_idx)
        lines.append(
            (
                get_cluster_id(clusters[a]),
                get_cluster_id(clusters[b]),
                dst[a, b],
                len(tuple(flatten_cluster(clusters[a]))),
            )
        )
        clusters[a] = new_cluster
        new_idx += 1
        clusters.pop(b)

    print(clusters)

    np.savetxt(output_file, np.array(lines))

    # import scipy.cluster.hierarchy as h
    # from scipy.spatial.distance import pdist
    # dst = pdist(df_np)
    # z = h.single(dst)
    # np.savetxt(output_file, z)

    # print(df)
