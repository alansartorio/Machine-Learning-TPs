import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

sns.set_theme()

# df = pl.read_csv('out/k_means_iterations_all_columns.csv')
df = pl.read_csv('out/k_means_iterations_numeric_columns.csv')

df = df.group_by(['run', 'k']).last()
print(df)

sns.lineplot(data = df, x = 'k', y = df['error'] / df['k'])
plt.show()

# print(df)
