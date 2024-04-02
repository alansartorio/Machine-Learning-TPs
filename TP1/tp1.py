import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'

sns.set_theme()

preferencias_britanicos = pd.read_csv("preferencias_britanicos.csv")
noticias_argentinas = pd.read_csv("noticias_argentinas.csv")
binary = pd.read_csv("binary.csv")

def laplace_correction(occurrences, total, classes_amount):
    return (occurrences + 1) / float(total + classes_amount)
    

def calculate_probability_given(df: pd.DataFrame, var: str, value: str, given_var: str, given_value: str,):
    in_class = df[df[given_var] == given_value]
    classes_amount = len(df[given_var].unique())
    occurrences = (in_class[var] == value).sum()
    total = len(in_class)
    return  laplace_correction(occurrences, total, classes_amount) 

def calculate_probability_of_being_in_class(var_probability: dict[str,dict[str,int]], class_probability:dict[str, int], values: dict[str, int], class_name: str):
    # P(clase/vars) = P(clase) * P(vars/clase) / P(vars)
    inverted_conditional = 1
    for var, value in values.items():
        p = var_probability[class_name][var]
        # P(A/C) = 1 - P(~A / C)
        if not value:
            p = 1 - p
        inverted_conditional *= p

    final_probability = class_probability[class_name] * inverted_conditional
    print(class_name, final_probability)
    return final_probability

def classify(var_probability: dict[str,dict[str,int]], class_probability: dict[str, int], values: dict[str, int]):
    classes = class_probability.keys()
    return max(classes, key=lambda c:calculate_probability_of_being_in_class(var_probability,class_probability,values, c))

def build_var_probability(df: pd.DataFrame, variables: list[str], class_var: str, class_values: list[str]):
    var_probability_given = dict()
    for class_name in class_values:
        var_probability_given[class_name] = dict()
        for var in variables:
            p_var = calculate_probability_given(df, var, 1, class_var, class_name)
            var_probability_given[class_name][var] = p_var
    return var_probability_given

def build_class_probability(df: pd.DataFrame, class_var: str, class_values: list[str]):
    class_probability = dict()
    for class_name in class_values:
        class_probability[class_name] = (df[class_var] == class_name).sum() / len(df)
    return class_probability

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
# print("Parte 2")
# print("Parte 3")
# part_3(binary)