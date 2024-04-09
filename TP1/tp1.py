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
    variables_and_dependencies = dict(
        gre=["rank"],
        gpa=["rank"],
        rank=["rank"], # Hmmmm
        admit=["gre", "gpa", "rank"]
    )

    vars_probability = dict(
        gre={(rank): 0 for rank in (1, 2, 3, 4)},
        gpa={(rank): 0 for rank in (1, 2, 3, 4)},
        rank={(rank): 0 for rank in (1, 2, 3, 4)}, 
        admit={(gre, gpa, rank): 0 for gre in (0, 1) for gpa in (0, 1) for rank in (1, 2, 3, 4)}
    )

    def calculate_probability_given_list_of_vars(df: pd.DataFrame, var: str, value: str, given_vars: list[str], given_values: list[str]):
        in_class = df
        for given_var, given_value in zip(given_vars, given_values):
            in_class = in_class[in_class[given_var] == given_value]
        classes_amount = len(df[given_vars].drop_duplicates())
        occurrences = (in_class[var] == value).sum()
        total = len(in_class)
        return  laplace_correction(occurrences, total, classes_amount)

    vars_probability['gre'][1] = calculate_probability_given_list_of_vars(clean_df, 'gre', 1, ['rank'], [1])
    vars_probability['gre'][2] = calculate_probability_given_list_of_vars(clean_df, 'gre', 1, ['rank'], [2])
    vars_probability['gre'][3] = calculate_probability_given_list_of_vars(clean_df, 'gre', 1, ['rank'], [3])
    vars_probability['gre'][4] = calculate_probability_given_list_of_vars(clean_df, 'gre', 1, ['rank'], [4])

    vars_probability['gpa'][1] = calculate_probability_given_list_of_vars(clean_df, 'gpa', 1, ['rank'], [1])
    vars_probability['gpa'][2] = calculate_probability_given_list_of_vars(clean_df, 'gpa', 1, ['rank'], [2])
    vars_probability['gpa'][3] = calculate_probability_given_list_of_vars(clean_df, 'gpa', 1, ['rank'], [3])
    vars_probability['gpa'][4] = calculate_probability_given_list_of_vars(clean_df, 'gpa', 1, ['rank'], [4])

    vars_probability['rank'] = build_class_probability(clean_df, 'rank', [1, 2, 3, 4])

    vars_probability['admit'][(0, 0, 1)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 0, 1])
    vars_probability['admit'][(0, 0, 2)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 0, 2])
    vars_probability['admit'][(0, 0, 3)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 0, 3])
    vars_probability['admit'][(0, 0, 4)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 0, 4])

    vars_probability['admit'][(1, 0, 1)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 0, 1])
    vars_probability['admit'][(1, 0, 2)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 0, 2])
    vars_probability['admit'][(1, 0, 3)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 0, 3])
    vars_probability['admit'][(1, 0, 4)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 0, 4])

    vars_probability['admit'][(0, 1, 1)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 1, 1])
    vars_probability['admit'][(0, 1, 2)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 1, 2])
    vars_probability['admit'][(0, 1, 3)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 1, 3])
    vars_probability['admit'][(0, 1, 4)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [0, 1, 4])

    vars_probability['admit'][(1, 1, 1)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 1, 1])
    vars_probability['admit'][(1, 1, 2)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 1, 2])
    vars_probability['admit'][(1, 1, 3)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 1, 3])
    vars_probability['admit'][(1, 1, 4)] = calculate_probability_given_list_of_vars(clean_df, 'admit', 1, ['gre', 'gpa', 'rank'], [1, 1, 4])
    
    # a) Probabilidad de que una persona que proviene de una escuela con rank 1 no sea admitida
    # P(admit=0 | rank=1) = P(admit=0, rank=1) / P(rank=1)

    print("a) ",vars_probability['admit'][(0, 0, 1)] * vars_probability['rank'][1])


    # b) Probabilidad de que una persona que proviene de una escuela con rank 2, GRE = 450 y GPA = 3.5 sea admitida
    # P(admit=1 | rank=2, gre=1, gpa=1) = P(admit, rank=2, gre=1, gpa=1) / P(rank=2, gre=1, gpa=1)
    # P(rank=2, gre=1, gpa=1) = P(rank=2) * P(gre=1 | rank=2) * P(gpa=1 | rank=2)

    print("b) ",vars_probability['admit'][(0, 1, 2)] / (vars_probability['rank'][2] * vars_probability['gre'][2] * vars_probability['gpa'][2]))

# print("Parte 1")
# part_1(preferencias_britanicos)
# print("Parte 2")
print("Parte 3")
part_3(binary)