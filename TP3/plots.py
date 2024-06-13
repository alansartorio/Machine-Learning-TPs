import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_theme()


def plot_step_function():
    x1 = np.linspace(-5, -0.01, 2)
    x2 = np.linspace(0.01, 5, 2)
    x = np.concatenate([x1, x2])
    y = np.sign(x)

    plt.plot(x, y)

    plt.savefig("plots/step_function.svg")
    plt.show()
    plt.cla()


plot_step_function()
