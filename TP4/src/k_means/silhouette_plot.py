if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import dataset
from dataset import DatasetType, load_dataset
from k_means import classify
import k_means
import numpy as np

sns.set_theme()

INPUT = "out/k_means/silhouette.csv"
OUTPUT = "plots/k_means/silhouette.svg"

df = pl.read_csv(INPUT)

sns.lineplot(data=df, x="k", y="silhouette")
plt.savefig(OUTPUT)
plt.show()
