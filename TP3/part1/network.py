
from abc import ABC
from functools import cached_property
import random
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .single_data import SingleData

FloatArray = npt.NDArray[np.float64]


class Layer:
    def __init__(self, weights: np.ndarray, activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.weights = weights
        self.activation_function = activation_function
        self.derivated_activation_function = derivated_activation_function

    @cached_property
    def perceptron_count(self):
        return self.weights.shape[0]

    @cached_property
    def previous_layer_perceptron_count(self):
        return self.weights.shape[1] - 1

    def propagate_without_activation(self, inputs):
        # return np.dot(self.weights, np.insert(inputs, 0, -1, axis=-1))
        inputs = np.insert(inputs, 0, -1, axis=0)
        return np.tensordot(self.weights, inputs, axes=((1,), (0,)))

    def calculate_previous_delta(self, previous_layer_h: FloatArray, output_delta: FloatArray):
        prod = sum(self.weights[i, 1:] * output_delta[i]
                   for i in range(self.perceptron_count))
        return self.derivated_activation_function(previous_layer_h) * prod


class EvaluationData:
    def __init__(self, h: Optional[FloatArray], v: FloatArray) -> None:
        self.h = h
        self.v = v



class Network(ABC):
    def __init__(self, weights: tuple[np.ndarray, ...], activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)) -> None:
        self.layers = tuple(Layer(layer_weights, activation_function, derivated_activation_function)
                            for layer_weights in weights)

    def layer_size(self, layer_index: int):
        return self.layers[layer_index].perceptron_count

    @cached_property
    def input_size(self):
        return self.layers[0].previous_layer_perceptron_count

    def calculate_vs_and_hs(self, inputs: FloatArray):
        values = [EvaluationData(None, inputs)]
        for layer in self.layers:
            current_h = layer.propagate_without_activation(values[-1].v)
            values.append(EvaluationData(
                current_h, layer.activation_function(current_h)))

        return values

    def evaluate(self, inputs: FloatArray):
        inputs = np.swapaxes(inputs, 0, -1)
        return np.swapaxes(self.calculate_vs_and_hs(inputs)[-1].v, 0, -1)

    def mean_squared_error(self, evaluation_data: Sequence[SingleData]):
        return self.error(evaluation_data) * 2 / len(evaluation_data)

    def error(self, evaluation_data: Sequence[SingleData]):
        return 0.5 * sum(np.sum((data.outputs - self.evaluate(data.inputs)) ** 2) for data in evaluation_data)

    def calculate_deltas(self, single_data: SingleData, evaluation_data: list[EvaluationData]):
        last_hs = evaluation_data[-1].h
        assert last_hs is not None, "Last layer Hs should not be None."
        last_vs = evaluation_data[-1].v
        last_layer_delta = self.layers[-1].derivated_activation_function(
            last_hs) * (single_data.outputs - last_vs)
        deltas = [last_layer_delta]

        for layer, layer_evalutation in reversed(list(zip(self.layers[1:], evaluation_data[1:]))):
            hs = layer_evalutation.h
            deltas.insert(0, layer.calculate_previous_delta(hs, deltas[0]))

        return deltas

    def train_single(self, learning_rate: float, single_data: SingleData):
        evaluation = self.calculate_vs_and_hs(single_data.inputs)
        deltas = self.calculate_deltas(single_data, evaluation)
        for m, layer in enumerate(self.layers):
            delta_w = learning_rate *\
                    np.dot(np.expand_dims(deltas[m], 1),
                       np.expand_dims(np.insert(evaluation[m].v, 0, -1), 0))
            # delta_w = [[learning_rate * d * v for v in evaluation[m].v]
                       # for d in deltas[m]]
            layer.weights = layer.weights + delta_w

    def train(self, learning_rate: float, train_data: Sequence[SingleData]):
        train_data = list(train_data)
        random.shuffle(train_data)
        for single_data in train_data:
            self.train_single(learning_rate, single_data)

        # self.train_single(learning_rate, random.sample(train_data, 1)[0])

    def randomize_weights(self, amplitude: float = 0.01):
        for layer in self.layers:
            layer.weights[:,:] = (np.random.rand(*layer.weights.shape) * 2 - 1) * amplitude

    @classmethod
    def with_zeroed_weights(cls, input_size: int, layers: tuple[int, ...], activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)):
        # + 1 to add threshold value
        return cls(tuple(np.zeros((current, previous + 1)) for previous, current in zip((input_size, ) + layers, layers)), activation_function, derivated_activation_function)


    @classmethod
    def with_random_weights(cls, input_size: int, layers: tuple[int, ...], activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)):
        m = cls.with_zeroed_weights(input_size, layers, activation_function, derivated_activation_function)
        m.randomize_weights()
        return m

    def copy(self):
        return Network(
            tuple(layer.weights.copy() for layer in self.layers),
            self.layers[0].activation_function,
            self.layers[0].derivated_activation_function
        )



class AutoEncoder(ABC):
    def __init__(self, weights: tuple[np.ndarray, ...], latent_layer_index: int, activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)) -> None:
        self.layers = tuple(Layer(layer_weights, activation_function, derivated_activation_function)
                            for layer_weights in weights)
        self.latent_layer_index = latent_layer_index

    def layer_size(self, layer_index: int):
        return self.layers[layer_index].perceptron_count

    @cached_property
    def input_size(self):
        return self.layers[0].previous_layer_perceptron_count

    def calculate_vs_and_hs(self, inputs: FloatArray):
        values = [EvaluationData(None, inputs)]
        for layer in self.layers:
            current_h = layer.propagate_without_activation(values[-1].v)
            values.append(EvaluationData(
                current_h, layer.activation_function(current_h)))

        return values

    def evaluate(self, inputs: FloatArray):
        inputs = np.swapaxes(inputs, 0, -1)
        return np.swapaxes(self.calculate_vs_and_hs(inputs)[-1].v, 0, -1)

    def mean_squared_error(self, evaluation_data: Sequence[SingleData]):
        return self.error(evaluation_data) * 2 / len(evaluation_data)

    def error(self, evaluation_data: Sequence[SingleData]):
        return 0.5 * sum(np.sum((data.outputs - self.evaluate(data.inputs)) ** 2) for data in evaluation_data)

    def calculate_deltas(self, single_data: SingleData, evaluation_data: list[EvaluationData]):
        last_hs = evaluation_data[-1].h
        assert last_hs is not None, "Last layer Hs should not be None."
        last_vs = evaluation_data[-1].v
        last_layer_delta = self.layers[-1].derivated_activation_function(
            last_hs) * (single_data.outputs - last_vs)
        deltas = [last_layer_delta]

        for layer, layer_evalutation in reversed(list(zip(self.layers[1:], evaluation_data[1:]))):
            hs = layer_evalutation.h
            deltas.insert(0, layer.calculate_previous_delta(hs, deltas[0]))

        return deltas

    def train_single(self, learning_rate: float, single_data: SingleData):
        evaluation = self.calculate_vs_and_hs(single_data.inputs)
        deltas = self.calculate_deltas(single_data, evaluation)
        for m, layer in enumerate(self.layers):
            delta_w = learning_rate *\
                np.dot(np.expand_dims(deltas[m], 1),
                       np.expand_dims(np.insert(evaluation[m].v, 0, -1), 0))
            # delta_w = [[learning_rate * d * v for v in evaluation[m].v]
            # for d in deltas[m]]
            layer.weights = layer.weights + delta_w

    def train(self, learning_rate: float, train_data: Sequence[SingleData]):
        train_data = list(train_data)
        random.shuffle(train_data)
        for single_data in train_data:
            self.train_single(learning_rate, single_data)

        # self.train_single(learning_rate, random.sample(train_data, 1)[0])

    def encode(self, inputs: FloatArray):
        inputs = np.swapaxes(inputs, 0, -1)
        return np.swapaxes(self.calculate_vs_and_hs(inputs)[self.latent_layer_index].v, 0, -1)

    def decode(self, latent: FloatArray):
        dec = AutoEncoder(tuple(layer.weights for layer in self.layers[self.latent_layer_index:]),
                          0, self.layers[0].activation_function, self.layers[0].derivated_activation_function)
        return dec.evaluate(latent)

    def randomize_weights(self, amplitude: float = 0.01):
        for layer in self.layers:
            layer.weights[:, :] = (np.random.rand(
                *layer.weights.shape) * 2 - 1) * amplitude

    @classmethod
    def with_zeroed_weights(cls, input_size: int, layers: tuple[int, ...], latent_layer_index: int, activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)):
        # + 1 to add threshold value
        return cls(tuple(np.zeros((current, previous + 1)) for previous, current in zip((input_size, ) + layers, layers)), latent_layer_index, activation_function, derivated_activation_function)

    @classmethod
    def with_random_weights(cls, input_size: int, layers: tuple[int, ...], latent_layer_index: int, activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)):
        m = cls.with_zeroed_weights(
            input_size, layers, latent_layer_index, activation_function, derivated_activation_function)
        m.randomize_weights()
        return m
