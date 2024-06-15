import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

sns.set_theme()

OUTPUT = "plots/k_means/error_by_k.svg"

df = pl.read_csv("out/k_means/iterations_all_columns.csv")

df = df.group_by(["run", "k"]).last()
print(df)

sns.lineplot(data=df, x="k", y=df["error"] / df["k"])

plt.savefig(OUTPUT)
plt.show()
