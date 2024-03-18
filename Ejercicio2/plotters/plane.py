import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'

sns.set_theme()

with open("data/data.json") as file:
    plane_data = json.load(file)

df = pd.read_csv("Advertising.csv", index_col=0)
df.columns = map(str.lower, df.columns)

# print(plane_data)
# print(df)

variables = [var['var'] for var in plane_data if var['var'] != 'beta_0']
values = [var['beta'] for var in plane_data if var['var'] != 'beta_0']

beta_0 = [var['beta'] for var in plane_data if var['var'] == 'beta_0']

assert len(variables) == 2

xx, yy = np.meshgrid([0, 300], [0, 50])

z = xx * values[0] + yy * values[1] + beta_0

# print(xx, yy, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z, alpha=0.2)
ax.scatter(df[variables[0]], df[variables[1]], df['sales'])

plt.show()
