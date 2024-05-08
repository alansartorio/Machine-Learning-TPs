import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import polars as pl

sns.set_theme()


def plot_confusion(df):
    confusion = df.group_by(['expected', 'predicted']).len()
    confusion = confusion.rows_by_key(['expected', 'predicted'])

    tp = confusion[(True, True)][0]
    tn = confusion[(False, False)][0]
    fp = confusion[(False, True)][0]
    fn = confusion[(True, False)][0]

    sns.heatmap(
        [[tn, fp], [fn, tp]],
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['True Negative', 'True Positive'],
        annot=True,
        cmap='Blues',
        fmt='0.0f',
    )

    plt.tight_layout()
    # plt.savefig('plots/part1_confusion.svg')
    plt.show()

df = pl.read_csv("out/part1_single_tree_results.csv")
plot_confusion(df)

def plot_precision_by_depth(df):
    confusion = df.group_by(['max depth', 'expected', 'predicted', 'data split']).len()
    confusion = confusion \
            .pivot(
                index=['max depth', 'data split'],
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

    sns.lineplot(confusion, x='max depth', y='precision', hue='data split')
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig('plots/part1_precision_over_depth.svg')
    # plt.show()

df = pl.read_csv("out/part1_results.csv")

plot_precision_by_depth(df)
