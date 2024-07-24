from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
sns.set_theme()


values="""Dataset,Training Accuracy,Test Accuracy
Original,0.95,0.97
PCA,0.94,0.94
KPCA,0.93,0.96
UMAP,0.96,0.98
Supervised UMAP,1.00,0.98"""
values = StringIO(values)

df = pd.read_csv(values)
df = pd.melt(df, id_vars=['Dataset'], value_vars=['Training Accuracy', 'Test Accuracy'])
print(df)

sns.barplot(data=df, hue='variable', x='Dataset', y='value')
plt.ylim((0.9, 1))
plt.savefig('plots/comparison.svg')

# plt.show()

