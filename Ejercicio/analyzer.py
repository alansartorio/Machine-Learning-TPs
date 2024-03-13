import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'

sns.set_theme()

df = pd.read_csv("Datos Alimenticios.csv", na_values='999.99', thousands=',')
df.rename(columns={
    "Grasas_sat": "Grasas Saturadas",
}, inplace=True)

print(df.columns)
figsize = (10, 10)

# 1) elimino datos faltantes
# df.dropna()

def saveplot(filename):
    def decorator(func):
        def wrapper():
            func()
            plt.tight_layout()
            plt.savefig(filename)
            if show_plots:
                plt.show()
            print(filename + " finished")
        return wrapper
    return decorator

@saveplot("part_2.svg")
def part_2():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=figsize)
    sns.boxplot(data=df, y='Grasas Saturadas', ax=ax1)
    sns.boxplot(data=df, y='Alcohol', ax=ax2)
    sns.boxplot(data=df, y='Calorías', ax=ax3)
    sns.countplot(data=df, x='Sexo', ax=ax4)

    ax4.bar_label(ax4.containers[0])

part_2()

@saveplot("part_3.svg")
def part_3():
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
    sns.boxplot(data=df, y='Grasas Saturadas', hue='Sexo', ax=ax1)
    sns.boxplot(data=df, y='Alcohol', hue='Sexo', ax=ax2)
    sns.boxplot(data=df, y='Calorías', hue='Sexo', ax=ax3)

part_3()

@saveplot("part_4.svg")
def part_4():
    plt.figure(figsize=figsize)
    df['Categoría'] = pd.cut(df['Calorías'], bins=[0, 1100, 1700, np.Infinity], include_lowest=True, labels=['0 - 1100', '1100 - 1700', '1700 - ∞'])

    # sns.scatterplot(data=df, x='Calorías', y='Alcohol', hue=df[['Categoría', 'Sexo']].apply(tuple, axis=1))
    # sns.scatterplot(data=df, x='Calorías', y='Alcohol', hue='Categoría')
    sns.boxplot(data=df, x='Categoría', y='Alcohol', hue='Sexo') #:like
    
part_4()
