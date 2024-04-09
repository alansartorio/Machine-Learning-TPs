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
    if class_amount is None:
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
    print(class_name, final_probability)
    return final_probability

def classify(var_probability: Dict[str,Dict[str,float]], class_probability: Dict[str, float], values: Dict[str, float]):
    classes = class_probability.keys()
    return max(classes, key=lambda c:calculate_probability_of_being_in_class(var_probability,class_probability,values, c))

def build_var_probability(df: pd.DataFrame, variables: List[str], class_var: str, class_values: List[str]):
    var_probability_given = dict()
    for class_name in class_values:
        var_probability_given[class_name] = dict()
        for var in variables:
            p_var = calculate_probability_given(df, var, 1, class_var, class_name)
            var_probability_given[class_name][var] = p_var
    return var_probability_given

def build_class_probability(df: pd.DataFrame, class_var: str, class_values: List[str]):
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

def classify_news(df: DataFrame, vocabulary: set) -> List[int]:
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
    df[vocab_vec_name] = df['data'].apply(build_vector)
    print('Vocabulary vectors built')

    print("Classifying documents...")
    return list(tqdm(map(lambda doc: classify(var_probabilities, class_probability, doc), df[vocab_vec_name])))


def part_1(df):
    classes = ('I', 'E')
    variables = ('scones','cerveza','wiskey','avena','futbol')
    class_name = 'Nacionalidad'

    var_probability = build_var_probability(df, variables, class_name, classes)
    class_probability = build_class_probability(df, 'Nacionalidad', classes)

    # a
    x1={'scones': 1, 'cerveza':0, 'wiskey': 1, 'avena':1, 'futbol':0}
    print("a)", x1, classify(var_probability, class_probability, x1))

    # b
    x2={'scones': 0, 'cerveza':1, 'wiskey': 1, 'avena':0, 'futbol':1}
    print("b)", x2, classify(var_probability, class_probability, x2))

def part_2(df):
    pre_process_data(df)
    df.dropna(inplace=True)
    vocabulary = get_vocabulary(df)
    training, evaluation = split_training_and_evaluation(df, 0.8)
    classification = classify_news(training, vocabulary)
    print(classification)


def part_3(df):
    def clean_data(df):
        df['gre'] = (df['gre'] >= 500).astype(int)
        df['gpa'] = (df['gpa'] >= 3).astype(int)
        return df

    clean_df = clean_data(df)
    classes = (0, 1)
    variables = ('gre', 'gpa', 'rank')
    class_name = 'admit'

    var_probability = build_var_probability(clean_df, variables, class_name, classes)
    class_probability = build_class_probability(clean_df, class_name, classes)

    # a -> 0.04479106692712246
    x1 = {'gre': 1, 'gpa': 1, 'rank': 1}
    print("a)", x1, classify(var_probability, class_probability, x1))

    # b -> 0.009660083505342415
    x2 = {'gre': 0, 'gpa': 1, 'rank': 2}
    print("b)", x2, classify(var_probability, class_probability, x2))



# print("Parte 1")
# part_1(preferencias_britanicos)
print("Parte 2")
part_2(noticias_argentinas)
# print("Parte 3")
# part_3(binary)