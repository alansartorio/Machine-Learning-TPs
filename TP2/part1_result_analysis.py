import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import polars as pl

sns.set_theme()

df = pl.read_csv("out/part1_results.csv")

print(df)
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
plt.savefig('plots/part1_confusion.svg')
# plt.show()
print(confusion)
