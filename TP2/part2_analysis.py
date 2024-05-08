from typing import Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

review_title, \
        review_text, \
        wordcount, \
        title_sentiment, \
        text_sentiment, \
        star_rating, \
        sentiment_value = \
        ("Review Title", "Review Text", "wordcount", "titleSentiment", "textSentiment", "Star Rating", "sentimentValue")
df = pd.read_csv("input/reviews_sentiment.csv", sep=";")

# CHECK: hay varias que tienen el title_sentiment vacio
df.drop(df[df[title_sentiment].isna()].index, inplace=True)

# CHECK: que valores le asignamos al negative y al positive?
df[title_sentiment] = df[title_sentiment].apply(lambda x:(-1, 1)[x=='positive'])

sns.set_theme()

show_plots = False

def avg_wordcount_per_star_rating():
    print("Average wordcount for 1 Star Rating", df[df[star_rating] == 1][wordcount].mean())

    ax = sns.barplot(df, x='Star Rating', y='wordcount')
    ax.set_title("Cantidad de palabras por Rating")

    if show_plots:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig('plots/part_2/wordcount_per_star_rating.svg')
    
    plt.clf()

def records_per_star_rating():
    ax = sns.histplot(df, x=star_rating)

    ax.set_title("Cantidad de datos por Rating")
    xticks = [1,2,3,4,5]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    if show_plots:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig('plots/part_2/count_per_star_rating.svg')

    plt.clf()

def graph_2d_wordcount_variable(varname, type = 'scatter'):
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    match type:
        case 'scatter':
            ax = sns.scatterplot(
                df, 
                x=wordcount, 
                y=varname, 
                hue=star_rating,
                palette=cmap 
                )
        case 'swarm':
            ax = sns.swarmplot(
                df, 
                x=wordcount, 
                y=varname, 
                hue=star_rating,
                palette=cmap 
                )
    ax.set_title(f"Distribución de puntos entre cantidad de palabras y {varname}")
    if show_plots:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(f'./plots/part_2/points_wordcount_{varname}_{type}.svg')

    plt.clf()

avg_wordcount_per_star_rating()
records_per_star_rating()
graph_2d_wordcount_variable(text_sentiment, 'swarm')
graph_2d_wordcount_variable(sentiment_value)

