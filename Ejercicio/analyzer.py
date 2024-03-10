import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()

df = pd.read_csv("Datos Alimenticios.csv", na_values='999.99', thousands=',')
df.rename(columns={
    "Grasas_sat": "Grasas Saturadas",
}, inplace=True)

print(df.columns)

# 1) elimino datos faltantes
df.dropna()

def part_3():
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    sns.boxplot(data=df, y='Grasas Saturadas', hue='Sexo', ax=ax1)
    sns.boxplot(data=df, y='Alcohol', hue='Sexo', ax=ax2)
    sns.boxplot(data=df, y='Calorías', hue='Sexo', ax=ax3)
    plt.show()
part_3()

def part_4():
    df['Categoría'] = pd.cut(df['Calorías'], bins=[0, 1100, 1700, np.Infinity], include_lowest=True, labels=['0 - 1100', '1100 - 1700', '1700 - ∞'])

    sns.scatterplot(data=df, x='Calorías', y='Alcohol', hue=df[['Categoría', 'Sexo']].apply(tuple, axis=1))
    # sns.boxplot(data=df, x='Categoría', y='Alcohol', hue='Sexo')
    plt.show()
part_4()
