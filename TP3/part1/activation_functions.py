from functools import partial
import numpy as np
from numba import njit
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@njit
def step_func(x: np.ndarray):
    return (x >= 0).astype(np.float64) * 2 - 1


@njit
def lineal_func(x: np.ndarray):
    return x


@njit
def sigmoid_tanh(b: float, x: float):
    return np.tanh(b * x)


@njit
def sigmoid_tanh_der(b: float, x: float):
    return 1 / np.cosh(b * x) * b


# @njit
def get_sigmoid_tanh(b: float):
    return lambda x: sigmoid_tanh(b, x), lambda x: sigmoid_tanh_der(b, x)


@njit
def get_sigmoid_tanh_1():
    return lambda x: sigmoid_tanh(1, x), lambda x: sigmoid_tanh_der(1, x)


@njit
def sigmoid_exp(b: float, x: np.ndarray):
    return 1 / (1 + np.exp(-2 * b * x))


@njit
def sigmoid_exp_der(b: float, x: np.ndarray):
    t = np.exp(-2 * b * x)
    return 2 * b * t / (1 + t) ** 2


# @njit
def get_sigmoid_exp(b: float):
    return lambda x: sigmoid_exp(b, x), lambda x: sigmoid_exp_der(b, x)


@njit
def get_sigmoid_exp_1():
    return lambda x: sigmoid_exp(1, x), lambda x: sigmoid_exp_der(1, x)


def softmax(x: FloatArray):
    e = np.exp(x)
    return e / np.sum(e, axis=-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.linspace(-5, 5, 200)
    f, d = get_sigmoid_tanh(1)
    plt.plot(X, f(X), label="tanh")
    plt.plot(X, d(X), label="dtanh")

    f, d = get_sigmoid_exp(1)
    plt.plot(X, f(X), label="exp")
    plt.plot(X, d(X), label="dexp")

    plt.legend()

    plt.show()
