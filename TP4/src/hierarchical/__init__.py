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

    df_np = df.select(numeric_vars).to_numpy()#[:20,:]
    # df_np = np.array([[0, 0], [0, 1], [0, 3]])
    count, vars = df_np.shape

    dst = np.zeros((count, count))

    for y in range(count):
        dst[y, y+1:] = np.linalg.norm(df_np[y+1:] - df_np[y], axis=1)
    print(dst)

    z = h.single(dst)

    import sys
    sys.setrecursionlimit(10000)

    truncate_cluster_count = 10
    with plt.rc_context({'lines.linewidth': 0.5}):
        h.dendrogram(z, truncate_cluster_count, truncate_mode='lastp', no_labels=True)
    plt.savefig(output_file)

    # dendo = ff.create_dendrogram(df_np)
    # dendo.write_image("plots/hierarchical/dendogram.svg")

    print(df)
