if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import plotly.figure_factory as ff
import scipy.cluster.hierarchy as h
import matplotlib.pyplot as plt
import sys
import numpy as np

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

lines = np.loadtxt(INPUT)

# z = h.single(dst)
sys.setrecursionlimit(10000)
truncate_cluster_count = 30  # 10
with plt.rc_context({"lines.linewidth": 0.5}):
    h.dendrogram(
        np.array(lines),
        truncate_cluster_count,
        truncate_mode="lastp",
        # no_labels=True,
    )
plt.savefig(OUTPUT)
plt.show()

# dendo = ff.create_dendrogram(df_np)
# dendo.write_image("plots/hierarchical/dendogram.svg")
