from itertools import product
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from typing import Callable, Dict, List, Tuple
import math
import random
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'

sns.set_theme()

preferencias_britanicos = pd.read_csv("preferencias_britanicos.csv")
noticias_argentinas = pd.read_csv("noticias_argentinas.csv")
binary = pd.read_csv("binary.csv")

def laplace_correction(occurrences, total, classes_amount):
    return (occurrences + 1) / float(total + classes_amount)
    
def calculate_predicate_probability_given(df: pd.DataFrame, predicate: Callable[[pd.Series], bool], given_var: str, given_value: str, class_amount = None, in_class_amount = None):
    in_class = df[df[given_var] == given_value]
    if class_amount is None:
        class_amount = len(df[given_var].unique())
    occurrences = in_class.apply(predicate, 'columns').sum()
    if in_class_amount is None:
        in_class_amount = len(in_class)
    return laplace_correction(occurrences, in_class_amount, class_amount) 

def calculate_probability_given(df: pd.DataFrame, var: str, value: str, given_var: str, given_value: str):
    return calculate_predicate_probability_given(df, lambda r:r[var] == value, given_var, given_value)

def calculate_probability_of_being_in_class(var_probability: Dict[str,Dict[str,float]], class_probability:Dict[str, float], values: Dict[str, float], class_name: str):
    # P(clase/vars) = P(clase) * P(vars/clase) / P(vars) -> P(vars) is not necessary if the class is the result
    inverted_conditional = 1
    for var, value in values.items():
        p = var_probability[class_name][var]
        # P(A/C) = 1 - P(~A / C)
        if not value:
            p = 1 - p
        inverted_conditional *= p

    p_vars = 0
    for clazz, prob in class_probability.items():
        p_vars_for_class = prob

        for var, value in values.items():
            p = var_probability[clazz][var]
            # P(A/C) = 1 - P(~A / C)
            if not value:
                p = 1 - p
            p_vars_for_class *= p
        p_vars += p_vars_for_class
    

    final_probability = class_probability[class_name] * inverted_conditional / p_vars
    # print(class_name, final_probability)
    return final_probability


def classify_and_get_probabilities(var_probability: dict[str,dict[str,float]], class_probability: dict[str, float], values: dict[str, float]):
    classes = class_probability.keys()
    probabilities = {c: calculate_probability_of_being_in_class(var_probability,class_probability,values, c) for c in classes}
    return max(classes, key=probabilities.__getitem__), probabilities

def classify(var_probability: dict[str,dict[str,float]], class_probability: dict[str, float], values: dict[str, float]):
    return classify_and_get_probabilities(var_probability, class_probability, values)[0]

def build_var_probability(df: pd.DataFrame, variables: List[str], class_var: str, class_values: List[str]):
    var_probability_given = dict()
    for class_name in class_values:
        var_probability_given[class_name] = dict()
        for var in variables:
            p_var = calculate_probability_given(df, var, 1, class_var, class_name)
            var_probability_given[class_name][var] = p_var
    return var_probability_given

def build_class_probability(df: pd.DataFrame, class_var: str, class_values: List[any]):
    class_probability = dict()
    for class_name in class_values:
        class_probability[class_name] = (df[class_var] == class_name).sum() / len(df)
    return class_probability

def split_training_and_evaluation(df: DataFrame, training_frac: float) -> Tuple[DataFrame, DataFrame]:
    training, evaluation = DataFrame(), DataFrame()
    for clazz in df['categoria'].unique():
        class_subset = df[df['categoria'] == clazz]
        training_mask =  np.random.rand(len(class_subset)) < training_frac
        training = pd.concat((training, class_subset[training_mask]))
        evaluation = pd.concat((evaluation, class_subset[~training_mask]))
    return (training, evaluation)

def get_cofusion_matrix(df: DataFrame):
    pass

def pre_process_data(df: DataFrame) -> DataFrame:
    def sanitize_title(s: str):
        # Remove symbols
        s = re.sub(r'[^\w]', ' ', s).lower()
        # Remove stop words
        # s = s
        # Tokenize string
        return set(v for v in s.split() if v)
    df['data'] = df['titular'].apply(sanitize_title)


def get_vocabulary(df: DataFrame):
    vocabulary = set()
    for vector in df['data']:
        vocabulary.update(vector)
    return vocabulary


# 3. Contar la cantidad de documentos de una clase que contienen a la palabra / la cantidad de documentos de esa clase
def get_word_probability(word: str, clazz: str, df: DataFrame, class_amount: int, in_class_amount: int):
    return calculate_predicate_probability_given(df, lambda r:word in r['data'], 'categoria', clazz, class_amount, in_class_amount)

def classify_news(df: DataFrame, evaluation: DataFrame, vocabulary: set) -> DataFrame:
    class_name = 'categoria'
    classes = set(df[class_name].tolist())
    print('Working with classes', classes)
    # var_probability = build_var_probability(df, vocabulary, class_name, classes)
    print('Computing class probabilities..')
    class_probability = build_class_probability(df, class_name, classes)
    print('Class probabilities built')
    # Build vocabulary vector
    vocab_vec_name = 'vector'
    vocabulary_list = list(vocabulary)
    def build_vector(words: List[str]) -> Dict[str, int]:
        base_vector = dict()
        for word in words:
            try:
                base_vector[word] = 1
            except:
                pass
        return base_vector
    
    var_probabilities = {}

    class_amount = len(classes)
    for clazz in tqdm(classes):
        var_probabilities[clazz] = {}
        in_class_amount = len(df[df['categoria'] == clazz])

        for word in tqdm(vocabulary_list):
            var_probabilities[clazz][word] = get_word_probability(word, clazz, df, class_amount, in_class_amount)
    
    print('Building the vocabulary vectors...')
    evaluation[vocab_vec_name] = evaluation['data'].apply(build_vector)
    print('Vocabulary vectors built')

    print("Classifying documents...")
    result = DataFrame()
    for index, doc in evaluation.iterrows():
        prediction, probabilities = classify_and_get_probabilities(var_probabilities, class_probability, doc[vocab_vec_name])
        tmp = DataFrame({'index': [index], 'expected': [doc['categoria']], 'prediction': [prediction], 'probability': [probabilities[prediction]]})
        result = pd.concat((result, tmp))
    result.set_index('index', inplace=True)

    return result

def part_1(df):
    classes = ('I', 'E')
    variables = ('scones','cerveza','wiskey','avena','futbol')
    class_name = 'Nacionalidad'

    var_probability = build_var_probability(df, variables, class_name, classes)
    class_probability = build_class_probability(df, 'Nacionalidad', classes)

    def plot(prob, name):
        res = DataFrame({f'P({"Ingles" if k == "I" else "Escoces"})': [v] for k, v in prob.items()})
        ax = sns.barplot(res)
        ax.bar_label(ax.containers[0])
        plt.savefig(f'plots/{name}.svg')
        plt.clf()


    # a
    x1={'scones': 1, 'cerveza':0, 'wiskey': 1, 'avena':1, 'futbol':0}
    class_x1, probabilities_x1 = classify_and_get_probabilities(var_probability, class_probability, x1)
    print("a)", x1, class_x1, probabilities_x1)

    plot(probabilities_x1, 'x1')
    

    # b
    x2={'scones': 0, 'cerveza':1, 'wiskey': 1, 'avena':0, 'futbol':1}
    class_x2, probabilities_x2 = classify_and_get_probabilities(var_probability, class_probability, x2)
    print("b)", x2, class_x2, probabilities_x2)

    plot(probabilities_x2, 'x2')

def run(args):
    df, vocabulary, split_frac = args
    training, evaluation = split_training_and_evaluation(df, split_frac)
    classification = classify_news(training, evaluation, vocabulary)
    classification.to_csv(f'data/split_{split_frac:.2}.csv')
    # print(classification)

def part_2(df):
    pre_process_data(df)
    df.dropna(inplace=True)
    # df = df.iloc[:1000]
    vocabulary = get_vocabulary(df)

    processes = 12
    with Pool(processes) as pool:
        splits = np.arange(0.2, 0.9, 0.1)
        assert len(splits) <= processes
        
        for _ in pool.imap_unordered(run, map(lambda split_frac:(df, vocabulary, split_frac), splits)):
            pass
        # for expected, predicted in zip(evaluation['categoria'], classification.iterrows()):
        #     print(expected, predicted)


def part_3(df):
    def clean_data(df):
        df['gre'] = (df['gre'] >= 500).astype(int)
        df['gpa'] = (df['gpa'] >= 3).astype(int)
        return df

    clean_df = clean_data(df)
    # clean_df.to_csv("cleaned.csv", index=False)

    vars_probability = dict(
        gre={},
        gpa={},
        rank={}, # Esta es categorica! Eso es un problema.
        admit={}
    )

    def calculate_probability_given_list_of_vars(df: pd.DataFrame, var: str, value: int, given_vars: list[str], given_values: list[int]):
        in_class = df
        for given_var, given_value in zip(given_vars, given_values):
            in_class = in_class[in_class[given_var] == given_value]
        classes_amount = len(df[given_vars].drop_duplicates())
        occurrences = (in_class[var] == value).sum()
        total = len(in_class)
        p = laplace_correction(occurrences, total, classes_amount)
        return {1: p, 0: 1 - p}

    parents = {
        'gre': ('rank',),
        'gpa': ('rank',),
        'rank': (),
        'admit': ('gre', 'gpa', 'rank'),
    }

    for a in (1, 2, 3, 4):
        vars_probability['gre'][(a,)] = calculate_probability_given_list_of_vars(clean_df, 'gre', 1, ['rank'], [a])
        vars_probability['gpa'][(a,)] = calculate_probability_given_list_of_vars(clean_df, 'gpa', 1, ['rank'], [a])

    vars_probability['rank'][()] = build_class_probability(clean_df, 'rank', [1, 2, 3, 4])

    for a in (0, 1):
        for b in (0, 1):
            for c in (1, 2, 3, 4):
                vars_probability['admit'][(a, b, c)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [a, b, c])
    

    print(vars_probability)

    def calculate_intersection_probability(vars_intersecting: list[str], values: list[int]):
        not_intersecting = [var for var in vars_probability.keys() if var not in vars_intersecting]
        # print(vars_intersecting, not_intersecting)

        # If the not_intersecting vars are not "rank" we need to see both cases, true and false
        var_possible_values = {
            "rank": (1, 2, 3, 4),
            "gre": (0, 1),
            "gpa": (0, 1),
            "admit": (0, 1),
        }
        acum = 0
        for vars in product(*[[(var, val) for val in var_possible_values[var]] for var in not_intersecting]):
            var_values = dict()
            for var, value in vars:
                var_values[var] = value
            for var, value in zip(vars_intersecting, values):
                var_values[var] = value
            # print(var_values)
            prob = 1
            for var in var_values.keys():
                var_parents = parents[var]
                parents_values = tuple(var_values[parent] for parent in var_parents)
                prob *= vars_probability[var][parents_values][var_values[var]]
            acum += prob

        return acum
            

    def plot_bar(v, filename):
        fig, ax = plt.subplots()
        ax.bar(0, v)
        ax.set_xlim((-1, 1))
        ax.set_ylim((0, 1))
        ax.bar_label(ax.containers[0])
        plt.savefig(f'plots/{filename}')
        plt.show()

    # a) Probabilidad de que una persona que proviene de una escuela con rank 1 no sea admitida
    # P(admit=0 | rank=1) = P(admit=0, rank=1) / P(rank=1)

    a = calculate_intersection_probability(['admit', 'rank'], [0, 1]) / calculate_intersection_probability(['rank'], [1])
    print(f"a) {a}")

    plot_bar(a, '3_a.svg')

    # print("a) ",
    #       (
    #           ((1 - vars_probability['admit'][(0, 0, 1)]) * (1-vars_probability['gre'])) + 
    #           (1 - vars_probability['admit'][(0, 1, 1)]) + 
    #           (1 - vars_probability['admit'][(1, 0, 1)]) + 
    #           (1 - vars_probability['admit'][(1, 1, 1)])
    #       ) / vars_probability['rank'][1])


    # b) Probabilidad de que una persona que proviene de una escuela con rank 2, GRE = 450 y GPA = 3.5 sea admitida
    # P(admit=1 | rank=2, gre=1, gpa=1) = P(admit, rank=2, gre=1, gpa=1) / P(rank=2, gre=1, gpa=1)
    b = calculate_intersection_probability(['admit', 'rank', 'gre', 'gpa'], [1, 2, 1, 1]) / calculate_intersection_probability(['rank', 'gre', 'gpa'], [2, 1, 1])
    print(f"b) {b}")

    plot_bar(b, '3_b.svg')

    # print("b) ", 
    #       vars_probability['admit'][(0, 1, 2)] / 
    #       (
    #           vars_probability['rank'][2] * 
    #           (1 - vars_probability['gre'][2]) * 
    #           vars_probability['gpa'][2])
    #     )

# print("Parte 1")
# part_1(preferencias_britanicos)
# print("Parte 2")
# part_2(noticias_argentinas)
print("Parte 3")
part_3(binary)
