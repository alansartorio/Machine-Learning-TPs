import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import polars as pl
import numpy as np

sns.set_theme()


def plot_confusion(df, filename):
    confusion = df.group_by(['expected', 'predicted']).len()
    confusion = confusion.rows_by_key(['expected', 'predicted'])

    tp = confusion[(True, True)][0]
    tn = confusion[(False, False)][0]
    fp = confusion[(False, True)][0]
    fn = confusion[(True, False)][0]


    conf = np.array([[tn, fp], [fn, tp]], dtype=np.float64)

    conf[0, :] = conf[0, :] / conf[0, :].sum() * 100
    conf[1, :] = conf[1, :] / conf[1, :].sum() * 100

    ax = sns.heatmap(
        conf,
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['True Negative', 'True Positive'],
        annot=True,
        cmap='Blues',
        fmt='0.1f',
        vmin=0,
        vmax=100
    )
    for t in ax.texts: t.set_text(t.get_text() + "%")

    plt.tight_layout()
    plt.savefig(f'plots/{filename}.svg')
    # plt.show()
    plt.clf()

def plot_precision_by_depth(single: pl.DataFrame, forest: pl.DataFrame):
    df = pl.concat([
        single.with_columns(pl.lit('single').alias('type')),
        forest.with_columns(pl.lit('forest').alias('type'))
    ])
    confusion = df.group_by(['type', 'max depth', 'expected', 'predicted', 'data split']).len()
    confusion = confusion \
            .pivot(
                index=['type', 'max depth', 'data split'],
                columns=['expected', 'predicted'],
                values='len',
            ) \
            .fill_null(0) \
            .rename({
                '{0,0}': 'tn',
                '{1,1}': 'tp',
                '{1,0}': 'fn',
                '{0,1}': 'fp',
            }) \
            .sort(by='max depth') \
            .with_columns([
                (pl.col('tp') / (pl.col('tp') + pl.col('fp'))).alias('precision'),
                (pl.col('tp') / (pl.col('tp') + pl.col('fn'))).alias('recall'),
            ])

    print(confusion)

    print(confusion.select(pl.concat_str([pl.col('type'), pl.col('data split')])).to_series())

    sns.lineplot(
        confusion,
        x='max depth',
        y='precision',
        hue='type',
        style='data split',
    )
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f'plots/part1_precision_over_depth.svg')
    # plt.show()
    plt.clf()


single = pl.read_csv("out/part1_single_tree_results.csv")
plot_confusion(
    single \
        .filter(pl.col('data split').eq('evaluation')) \
        .filter(pl.col('max depth').eq(4)),
    'part1_single_confusion'
)


forest = pl.read_csv("out/part1_results.csv")
print(forest)
plot_confusion(
    forest \
        .filter(pl.col('data split').eq('evaluation'))
        .filter(pl.col('max depth').eq(5))
        .filter(pl.col('tree count').eq(8))
        .filter(pl.col('bag size').eq(256)),
    'part1_forest_confusion'
)

plot_precision_by_depth(single, forest)
