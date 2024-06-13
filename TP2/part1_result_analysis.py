import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import polars as pl
import numpy as np

sns.set_theme()

best_tree_count = 16
best_subset_size = 128
best_tree_depth = 4


def plot_confusion(df, filename):
    confusion = df.group_by(["expected", "predicted"]).len()
    confusion = confusion.rows_by_key(["expected", "predicted"])

    tp = confusion[(True, True)][0]
    tn = confusion[(False, False)][0]
    fp = confusion[(False, True)][0]
    fn = confusion[(True, False)][0]

    conf = np.array([[tn, fp], [fn, tp]], dtype=np.float64)

    conf[0, :] = conf[0, :] / conf[0, :].sum() * 100
    conf[1, :] = conf[1, :] / conf[1, :].sum() * 100

    ax = sns.heatmap(
        conf,
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["True Negative", "True Positive"],
        annot=True,
        cmap="Blues",
        fmt="0.1f",
        vmin=0,
        vmax=100,
    )
    for t in ax.texts:
        t.set_text(t.get_text() + "%")

    plt.tight_layout()
    plt.savefig(f"plots/{filename}.svg")
    # plt.show()
    plt.clf()


def plot_precision_by_depth(single: pl.DataFrame, forest: pl.DataFrame):
    df = pl.concat(
        [
            single.with_columns(pl.lit("single").alias("type")),
            forest.with_columns(pl.lit("forest").alias("type")),
        ],
        how="diagonal",
    )
    print(df)
    confusion = df.group_by(df.columns).len()
    other_columns = set(confusion.columns) - {"expected", "predicted", "len"}
    print(other_columns)
    confusion = (
        confusion.pivot(
            index=list(other_columns),
            columns=["expected", "predicted"],
            values="len",
        )
        .fill_null(0)
        .rename(
            {
                "{0,0}": "tn",
                "{1,1}": "tp",
                "{1,0}": "fn",
                "{0,1}": "fp",
            }
        )
        .sort(by="max depth")
        .with_columns(
            [
                (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).alias("precision"),
                (pl.col("tp") / (pl.col("tp") + pl.col("fn"))).alias("recall"),
            ]
        )
    )

    # with pl.Config(tbl_rows=50, tbl_cols=50, fmt_str_lengths=100):
    # print(confusion.filter(pl.col('bag size').eq(677)))

    def plot_precision(confusion, figure, x):
        with pl.Config(tbl_rows=50, tbl_cols=50, fmt_str_lengths=100):
            print(confusion)
        sns.lineplot(
            confusion,
            x=x,
            y="precision",
            hue="type",
            style="data split",
            markers="o",
        )
        plt.ylim((0, 1))
        plt.tight_layout()
        plt.savefig(f"plots/part1{figure}_precision_over_{x}.svg")
        plt.show()
        plt.clf()

    print(
        "BEST",
        confusion.filter(
            pl.col("data split").eq("evaluation"),
            pl.col("type").eq("forest"),
            pl.col("max depth").eq(best_tree_depth),
            pl.col("bag size").eq(best_subset_size),
        )
        .filter(pl.col("precision") == pl.col("precision").max())
        .select(pl.col("tree count")),
    )

    print(
        "BEST",
        confusion.filter(
            pl.col("data split").eq("evaluation"),
            pl.col("type").eq("forest"),
            pl.col("max depth").eq(best_tree_depth),
            pl.col("tree count").eq(best_tree_count),
        )
        .filter(pl.col("precision") == pl.col("precision").max())
        .select(pl.col("bag size")),
    )

    print(
        "BEST",
        confusion.filter(
            pl.col("data split").eq("evaluation"),
            pl.col("type").eq("forest"),
            pl.col("bag size").eq(best_subset_size),
            pl.col("tree count").eq(best_tree_count),
        )
        .filter(pl.col("precision") == pl.col("precision").max())
        .select(pl.col("max depth")),
    )

    plot_precision(
        confusion.filter(
            pl.col("tree count").eq(best_tree_count)
            & pl.col("max depth").eq(best_tree_depth)
            & pl.col("type").eq("forest")
        ),
        "",
        x="bag size",
    )

    plot_precision(
        confusion.filter(
            pl.col("bag size").eq(best_subset_size)
            & pl.col("max depth").eq(best_tree_depth)
            & pl.col("type").eq("forest")
        ),
        "",
        x="tree count",
    )
    plot_precision(
        confusion.filter(
            (
                pl.col("tree count").eq(best_tree_count)
                & pl.col("bag size").eq(best_subset_size)
                & pl.col("type").eq("forest")
            )
            | pl.col("type").eq("single")
        ),
        "_single_vs_forest",
        x="max depth",
    )
    print(
        confusion.select(
            pl.concat_str([pl.col("type"), pl.col("data split")])
        ).to_series()
    )


single = pl.read_csv("out/part1_single_tree_results.csv")
plot_confusion(
    single.filter(pl.col("data split").eq("evaluation")).filter(
        pl.col("max depth").eq(4)
    ),
    "part1_single_confusion",
)


forest = pl.read_csv("out/part1_results.csv")
print(forest)
plot_confusion(
    forest.filter(pl.col("data split").eq("evaluation"))
    .filter(pl.col("max depth").eq(best_tree_depth))
    .filter(pl.col("tree count").eq(best_tree_count))
    .filter(pl.col("bag size").eq(best_subset_size)),
    "part1_forest_confusion",
)

# sns.lineplot(forest, x='bag size', y='')
plot_precision_by_depth(single, forest)
