import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

df = pd.read_csv('out/sorted.csv')

def evaluate_k(k):
    successful = 0
    total = 0

    for idx, group in df.groupby('original index'):
        expected = group['expected result'].iloc[0]
        actual = group.iloc[:k]['Star Rating'].mode()[0]
        # print('expected', expected)
        # print('actual', actual)
        successful += actual == expected
        total += 1

    return successful, total

max_k = df['index'].max()

data = pd.DataFrame()
for k in range(1, max_k + 1, 2):
    success, total = evaluate_k(k)
    data = pd.concat([data, pd.DataFrame({
        'k': [k],
        'success %': [success / total]
    })])

data.reset_index(drop=True, inplace=True)

ax = sns.barplot(data=data, x='k', y='success %')
ax.set_ylim((0, 1))
plt.show()
print(data)
