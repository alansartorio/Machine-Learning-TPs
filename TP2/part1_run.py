import polars as pl
from polars import DataFrame
import tp2
import numpy as np
import json
from typing import Any, Callable, Dict, Optional, Tuple
from part1_fetch import get_data, creditability, credit_amount, age, duration_of_credit
import logging
from multiprocessing import Pool
from tqdm import tqdm

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

def print_unique_ns(df):
    unique_values = {column.name: column.n_unique() for column in df.get_columns()}
    print(json.dumps(unique_values, indent=4))

class Forest:
    def __init__(self) -> None:
        self.trees = []

    def train(self, tree_count: int, subset_generator: Callable[[int], DataFrame], value_mapping: Dict[str, Dict[int, str]], max_depth: Optional[int] = None):
        for tree_index in range(tree_count):
            subset = subset_generator(tree_index)
            tree = tp2.train(subset, creditability, value_mapping, max_depth)
            self.trees.append(tree)

    def classify(self, row):
        del row[creditability]
        results = pl.Series([tree.classify(row) for tree in self.trees])
        # print(results)
        mode = results.mode()[0]
        # print(mode)
        return mode
        

def train_model(df: pl.DataFrame,value_mapping: Dict[str, Dict[int, str]], subset_size: Optional[int] = None, tree_count: int = 1, max_depth: Optional[int] = None) -> Forest:
    print_unique_ns(df)
    # print(json.dumps(value_mapping, indent=4))
    # print(df)

    forest = Forest()
    # print('Len', len(df))

    if subset_size is None:
        subset_chooser = lambda _:df
    else:
        subset_chooser = lambda i:df.sample(400, with_replacement=True, seed=i * 12412)
    forest.train(tree_count, subset_chooser, value_mapping, max_depth=max_depth)

    return forest

def evaluate_model(df: DataFrame, forest: Forest) -> DataFrame:
    results = []

    for row in df.iter_rows(named = True):
        expected = row[creditability]
        # del row[creditability]
        result = forest.classify(row)
        results.append({"expected": expected, "predicted": result})
    results = DataFrame(results)
    return results
    # results.write_csv("out/part1_results.csv")

def train(arg):
    # print(arg)
    training, value_mapping, tree_count, subset_size, tree_depth = arg
    # print(f'Training with tree count = {tree_count}, subset_size = {subset_size}, tree depth = {tree_depth}')
    return train_model(training, value_mapping, subset_size=subset_size, tree_count=tree_count, max_depth=tree_depth)
# train = lambda (tree_count, subset_size, tree_depth): train_model(training, value_mapping, subset_size=subset_size, tree_count=tree_count, max_depth=tree_depth)
        

if __name__ == '__main__':
    df = get_data()
    value_mapping = {column.name: {value:str(value) for value in column.unique()} for column in df.get_columns()}

    def reduce_column(df: pl.DataFrame, column, factor):
        series: pl.Series = (df.get_column(column) / factor).round().cast(pl.Int64)
        value_mapping[column] = {reduced: f'[{reduced * factor}, {(reduced + 1) * factor})' for reduced in series.unique()}
        return df.with_columns(series.alias(column))

    df = reduce_column(df, credit_amount, 2000)
    df = reduce_column(df, age, 5)
    df = reduce_column(df, duration_of_credit, 10)
    df = df.with_columns(df.get_column(creditability).cast(pl.String))

    def split_training_and_evaluation(df: DataFrame, training_frac: float) -> Tuple[DataFrame, DataFrame]:
        training, evaluation = DataFrame(), DataFrame()
        np.random.seed(seed=13123414)
        training_mask =  np.random.rand(len(df)) < training_frac
        training = pl.concat((training, df.filter(training_mask)))
        evaluation = pl.concat((evaluation, df.filter(~training_mask)))
        return (training, evaluation)

    training, evaluation = split_training_and_evaluation(df, 0.7)

    print_unique_ns(df)
    value_mapping['Creditability'] = {}

    def single_tree():
        all_results = []
        for tree_depth in range(1, 11):
            print(f'Training with tree depth = {tree_depth}')
            forest = train_model(training, value_mapping, subset_size=None, tree_count=1, max_depth=tree_depth)
            print()
            evaluation_results = evaluate_model(evaluation, forest).with_columns(pl.lit("evaluation").alias("data split"))
            training_results = evaluate_model(training, forest).with_columns(pl.lit("training").alias("data split"))
            results = pl.concat([evaluation_results, training_results]).with_columns(pl.lit(tree_depth).alias("max depth"))
            all_results.append(results)
            if tree_depth == 1:
                with open("out/single_tree_depth_1.dot", 'w') as graph_file:
                    print(forest.trees[0].to_graphviz(), file=graph_file)


        all_results = pl.concat(all_results, rechunk=True)\
        .with_columns([
            pl.lit(1).alias("tree count"),
            pl.lit(len(training)).alias("bag size")
        ])

        all_results.write_csv("out/part1_single_tree_results.csv")

    # single_tree()

    def single_tree_for_graph():
        forest = train_model(training, value_mapping, subset_size=None, tree_count=1, max_depth=2)
        # results = evaluate_model(evaluation, forest)
        # results.write_csv("out/part1_single_tree_results.csv")
        
        print('Tree count', len(forest.trees))
        with open("out/single_tree_depth_2.dot", 'w') as graph_file:
            print(forest.trees[0].to_graphviz(), file=graph_file)

    # single_tree_for_graph()

    tree_counts = (2, 4, 8, 16, 32)
    subset_sizes = (32, 64, 128, 256, 512)
    tree_depths = tuple(range(1, 11))

    best_tree_count = 16
    best_subset_size = 128
    best_tree_depth = 4

    def forest():
        all_results = []
        configs = set()
        for tree_count in tree_counts:
            configs.add((tree_count, best_subset_size, best_tree_depth))
        for subset_size in subset_sizes:
            configs.add((best_tree_count, subset_size, best_tree_depth))
        for tree_depth in tree_depths:
            configs.add((best_tree_count, best_subset_size, tree_depth))
        configs = list(configs)

        complete_configs = list((training.clone(), value_mapping.copy(), *config) for config in configs)
        # print(complete_configs)
        main_forest = None

        # with Pool(12) as pool:
        for forest, (tree_count, subset_size, tree_depth) in tqdm(zip(map(train, complete_configs), configs), total=len(configs)):
            # forest = train_model(training, value_mapping, subset_size=subset_size, tree_count=tree_count, max_depth=tree_depth)
            # print()
            if tree_count == best_tree_count and \
                subset_size == best_subset_size and \
                tree_depth == best_tree_depth:
                main_forest = forest
            evaluation_results = evaluate_model(evaluation, forest).with_columns(pl.lit("evaluation").alias("data split"))
            training_results = evaluate_model(training, forest).with_columns(pl.lit("training").alias("data split"))
            results = pl \
                .concat([evaluation_results, training_results]) \
                .with_columns([
                    pl.lit(tree_depth).alias("max depth"),
                    pl.lit(tree_count).alias("tree count"),
                    pl.lit(subset_size).alias("bag size")
                ])
            all_results.append(results)

        first_attributes = [tree.get_first_split() for tree in main_forest.trees]
        DataFrame({'first attribute': first_attributes}).write_csv("out/part1_first_attributes.csv")

        all_results = pl.concat(all_results, rechunk=True)

        all_results.write_csv("out/part1_results.csv")
        
    forest()
    
    # # print('Tree count', len(forest.trees))
    # with open("out/single_tree.dot", 'w') as graph_file:
        # print(forest.trees[0].to_graphviz(), file=graph_file)
