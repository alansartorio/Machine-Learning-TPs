import polars as pl
import tp2
import json
from typing import Dict
from part1_fetch import get_data, creditability, credit_amount, age, duration_of_credit

def print_unique_ns(df):
        unique_values = {column.name: column.n_unique() for column in df.get_columns()}
        print(json.dumps(unique_values, indent=4))

def train_model(df: pl.DataFrame,value_mapping: Dict[str, Dict[int, str]]):
    def reduce_column(df: pl.DataFrame, column, factor):
        series: pl.Series = (df.get_column(column) / factor).round().cast(pl.Int64)
        value_mapping[column] = {reduced: f'[{reduced * factor}, {(reduced + 1) * factor})' for reduced in series.unique()}
        return df.with_columns(series.alias(column))

    df = reduce_column(df, credit_amount, 2000)
    df = reduce_column(df, age, 5)
    df = reduce_column(df, duration_of_credit, 10)
    df = df.with_columns(df.get_column(creditability).cast(pl.String))


    print_unique_ns(df)
    print(json.dumps(value_mapping, indent=4))
    exit()
    print(df)

    tree = tp2.train(df, creditability, value_mapping)
    with open("tree.dot", 'w') as graph_file:
        print(tree.to_graphviz(), file=graph_file)

    for row in df.iter_rows(named = True):
        expected = row[creditability]
        del row[creditability]
        result = tree.classify(row)
        if expected != result:
            print(expected, result)

if __name__ == '__main__':
    df = get_data()
    value_mapping = {column.name: {value:str(value) for value in column.unique()} for column in df.get_columns()}
    print_unique_ns(df)
    value_mapping['Creditability'] = {}
    train_model(df, value_mapping)
