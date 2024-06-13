from typing import Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np

(
    review_title,
    review_text,
    wordcount,
    title_sentiment,
    text_sentiment,
    star_rating,
    sentiment_value,
) = (
    "Review Title",
    "Review Text",
    "wordcount",
    "titleSentiment",
    "textSentiment",
    "Star Rating",
    "sentimentValue",
)
df = pd.read_csv("input/reviews_sentiment.csv", sep=";")

# CHECK: hay varias que tienen el title_sentiment vacio
df.drop(df[df[title_sentiment].isna()].index, inplace=True)

# CHECK: que valores le asignamos al negative y al positive?
df[title_sentiment] = df[title_sentiment].apply(lambda x: (-1, 1)[x == "positive"])


def split_training_and_evaluation(
    df: DataFrame, training_frac: float
) -> Tuple[DataFrame, DataFrame]:
    training, evaluation = DataFrame(), DataFrame()
    np.random.seed(11241241)
    training_mask = np.random.rand(len(df)) < training_frac
    training = pd.concat((training, df[training_mask]))
    evaluation = pd.concat((evaluation, df[~training_mask]))
    return (training, evaluation)


training, test = split_training_and_evaluation(df, 0.8)

input_columns = (wordcount, title_sentiment, sentiment_value)


# print(test[list(input_columns)].to_string())
def dist(a, b):
    to_array = lambda row: np.array(list(row[col] for col in input_columns))
    return np.linalg.norm(to_array(a) - to_array(b))


def sort_by_distance(test_row, training: DataFrame):
    copy = training.copy()
    copy["distance"] = copy.apply(lambda row: dist(test_row, row), axis=1)
    copy.sort_values(by="distance", inplace=True)
    return copy


complete = DataFrame()

for idx, row in test.iterrows():
    sorted = sort_by_distance(row, training)
    sorted.reset_index(drop=True, inplace=True)
    sorted.reset_index(inplace=True)
    sorted.drop(
        [
            review_title,
            review_text,
            wordcount,
            title_sentiment,
            text_sentiment,
            sentiment_value,
        ],
        axis="columns",
        inplace=True,
    )
    sorted["original index"] = idx
    sorted["expected result"] = row[star_rating]
    complete = pd.concat([complete, sorted], ignore_index=True)

print(complete)
complete.to_csv("out/sorted.csv", index=False)
