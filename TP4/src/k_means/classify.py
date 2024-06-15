if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import dataset
from dataset import DatasetType, load_dataset
from k_means import classify
import k_means
import numpy as np

sns.set_theme()

OUTPUT="plots/k_means/classification.svg"

run = 10
k = 5

centroids = (
    pl.read_csv("out/k_means/centroids_all_columns.csv")
    .filter((pl.col("k") == k) & (pl.col("run") == run))
    .drop("k", "run")
)


df_normalized = load_dataset(DatasetType.NORMALIZED)
df_numerical = load_dataset(DatasetType.NUMERICAL)

# np.set_printoptions(threshold=sys.maxsize)
clusters = classify(df_normalized.select(k_means.all_numeric_columns), centroids)

categorized = df_numerical.with_columns(pl.Series(clusters).alias("cluster")).filter(
    pl.col(dataset.genres).is_in(("Action", "Comedy", "Drama"))
)

categorized = categorized.pivot(
    index=dataset.genres,
    columns="cluster",
    values=dataset.imdb_id,
    aggregate_function=pl.len(),
    sort_columns=True,
).fill_null(0)


def scale_to_sum_1(values):
    values = np.array(values)
    return tuple(values / values.sum() * 100)


categorized = categorized.select(
    dataset.genres,
    values=pl.concat_list(pl.all().exclude(dataset.genres))
    .map_elements(scale_to_sum_1, return_dtype=pl.List(pl.Float64))
    .list.to_struct(fields=str),
).unnest("values")

ax = sns.heatmap(
    categorized.to_pandas().set_index(dataset.genres),
    annot=True,
    cmap="Blues",
    fmt="0.1f",
    vmin=0,
    vmax=100,
    yticklabels=True,
)
for t in ax.texts:
    t.set_text(t.get_text() + "%")
ax.set_ylabel("Genre")
ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig(OUTPUT)
plt.show()
