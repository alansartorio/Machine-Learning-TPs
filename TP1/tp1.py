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

def part_1(df):
    var_probability_given = dict()
    classes = ('I', 'E')
    variables = ('scones','cerveza','wiskey','avena','futbol')
    for class_name in classes:
        var_probability_given[class_name] = dict()
        for var in variables:
            p_var = calculate_probability_given(df, var, 1, 'Nacionalidad', class_name)
            var_probability_given[class_name][var] = p_var

    class_probability = dict()
    for class_name in classes:
        class_probability[class_name] = (df['Nacionalidad'] == class_name).sum() / len(df)

    # var_probability = dict()
    # for var in variables:
    #     var_probability[var] = (df[var] == 1).sum() / len(df)

    print(var_probability_given)

    def calculate_probability_of_being_in_class(values: dict[str, int], class_name: str):
        # P(clase/vars) = P(clase) * P(vars/clase) / P(vars)
        inverted_conditional = 1
        for var, value in values.items():
            p = var_probability_given[class_name][var]
            # P(A/C) = 1 - P(~A / C)
            if not value:
                p = 1 - p
            inverted_conditional *= p

        final_probability = class_probability[class_name] * inverted_conditional
        print(final_probability)
        return final_probability

    def classify(values: dict[str, int]):
        return max(classes, key=lambda c:calculate_probability_of_being_in_class(values, c))

    x1={'scones': 1, 'cerveza':0, 'wiskey': 1, 'avena':1, 'futbol':0}
    print(x1, classify(x1))
    x2={'scones': 0, 'cerveza':1, 'wiskey': 1, 'avena':0, 'futbol':1}
    print(x2, classify(x2))

part_1(preferencias_britanicos)