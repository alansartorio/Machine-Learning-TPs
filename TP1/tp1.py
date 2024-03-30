import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'

sns.set_theme()


binary = pd.read_csv("binary.csv")
preferencias_britanicos = pd.read_csv("preferencias_britanicos.csv")
noticias_argentinas = pd.read_csv("noticias_argentinas.csv")

print(binary)
print(preferencias_britanicos)
print(noticias_argentinas)