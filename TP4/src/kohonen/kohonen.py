import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Union
import seaborn as sns
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import polars as pl

from config import Input, load_config, KohonenConfig
from radius_update import ProgressiveReduction, RadiusUpdate
from similarity import EuclideanSimilarity, Similarity
from standardization import ZScore

import sys

# setting path
sys.path.append("src")
import dataset
from analisys import PlotConfig


class KohonenNetwork:

    def __init__(
        self,
        k: int,
        r: float,
        input_size: int,
        learning_rate: float,
        max_epochs: int,
        initial_input: Optional[list] = None,
        radius_update: RadiusUpdate = ProgressiveReduction(),
        similarity: Similarity = EuclideanSimilarity(),
    ):
        self.input = np.zeros(input_size)
        self.input_size = input_size
        self.k = k
        self.radius = r
        self.original_radius = r
        self.radius_update = radius_update
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.similarity = similarity

        self.weights = self.initialize_weights(k, input_size, initial_input)

    @staticmethod
    def initialize_weights(
        k: int, input_size: int, weights: Optional[list]
    ) -> np.ndarray:
        if weights is None:
            return np.random.uniform(0, 1, k**2 * input_size).reshape(
                (k**2, input_size)
            )
        return np.array(weights).reshape((k**2, input_size))

    def train(self, data: np.ndarray):
        for current_epoch in range(0, self.max_epochs):

            self.input = data[np.random.randint(0, len(data))]

            winner_neuron = self.get_winner(self.input)
            self.update_weights(winner_neuron)

            self.radius = self.radius_update.update(self.original_radius, current_epoch)

    def update_weights(self, neuron):
        neighbourhood = self.get_neighbourhood(neuron)
        for i in neighbourhood:
            self.weights[i] = self.weights[i] + self.learning_rate * (
                self.input - self.weights[i]
            )

    def get_neighbourhood(self, neuron_index: int) -> list:
        """given the index of a neuron, returns an array of all the neighbouring neurons
        inside the current radius"""
        neighbours = []
        matrix_shape = (self.k, self.k)
        neuron_coords_array = np.array(np.unravel_index(neuron_index, matrix_shape))
        for i in range(0, self.k**2):
            coord = np.unravel_index(i, matrix_shape)
            distance = np.linalg.norm(neuron_coords_array - np.array(coord))
            if distance <= self.radius:
                neighbours.append(i)
        return neighbours

    def get_winner(self, input_data) -> int:
        weights_size = self.k**2
        distances = np.zeros(weights_size)
        for i in range(weights_size):
            distances[i] = self.similarity.calculate(input_data, self.weights[i])

        return int(np.argmin(distances))

    def __get_direct_neighbours_coords(self, index: int):
        matrix_shape = (self.k, self.k)
        coord = np.unravel_index(index, matrix_shape)
        neighbours = []
        possible_neighbours = [
            (coord[0], coord[1] + 1),
            (coord[0], coord[1] - 1),
            (coord[0] + 1, coord[1]),
            (coord[0] - 1, coord[1]),
        ]
        for p in possible_neighbours:
            if 0 <= p[0] <= self.k - 1 and 0 <= p[1] <= self.k - 1:
                neighbours.append(np.ravel_multi_index(p, matrix_shape))
        return neighbours

    def get_unified_distance_matrix(self):
        matrix = np.zeros((self.k, self.k))

        for i in range(self.k**2):
            neighbours = self.__get_direct_neighbours_coords(i)
            i_coord = np.unravel_index(i, (self.k, self.k))
            matrix[i_coord] = np.mean(
                list(
                    map(
                        lambda n: np.linalg.norm(self.weights[i] - self.weights[n]),
                        neighbours,
                    )
                )
            )
        matrix = [matrix[i : i + self.k] for i in range(0, len(matrix), self.k)][::-1]

        return [item for row in matrix for item in row]


def save_result(
    config: KohonenConfig,
    dataset_type: dataset.DatasetType,
    inputs: Input,
    kohonen: KohonenNetwork,
    genre_groups: List[Counter[str]],
    input_size: int,
) -> None:
    output_directory = Path("out", "data")
    os.makedirs(output_directory, exist_ok=True)

    with open(
        output_directory.joinpath(f"result-{datetime.now()}.json"),
        "w",
        encoding="utf-8",
    ) as file:
        result = {
            "config": config.to_json(),
            "dataset": {
                "name": dataset_type.value.get("path"),
                "input_size": input_size,
                "size": len(inputs),
            },
            "groups": genre_groups,
            "weights": kohonen.weights.tolist(),
        }
        json.dump(result, file, ensure_ascii=False, indent=4)


def plot_u_matrix(kohonen: KohonenNetwork, plot_config: PlotConfig = PlotConfig()):
    u_matrix = kohonen.get_unified_distance_matrix()
    u_matrix = np.flip(u_matrix, axis=0)

    plt.title("Unified Distance Matrix Heatmap")

    sns.heatmap(u_matrix, cmap="gray", annot=True)

    plot_config.print_plot()
    plt.clf()


def plot_neuron_classification(
    genre_activations: List[Counter[str]],
    genres: List[str],
    network_config: KohonenConfig,
    plot_config: PlotConfig = PlotConfig(),
) -> None:
    K = network_config.grid_size
    R = network_config.neighbours_radius
    LEARNING_RATE = network_config.learning_rate
    MAX_EPOCHS = network_config.epochs

    def neuron_display_name(neuron_activations: Counter[str]) -> str:
        return neuron_activations.most_common(1)[0][0]

    plt.figure(figsize=plot_config.fig_size if plot_config.fig_size else (10, 8))
    plt.title(
        f"Claisificación por neurona. Red {K}x{K} con η(0)={str(LEARNING_RATE)}, R={str(R)} y {MAX_EPOCHS} epochs"
    )

    network_shape = (K, K)
    groups = np.zeros(network_shape, dtype=int)
    for neuron_index, genre_activations_in_neuron in enumerate(genre_activations):
        winner_genre = neuron_display_name(genre_activations_in_neuron)
        i, j = np.unravel_index(neuron_index, network_shape)
        plt.text(
            j + 0.5,
            K - 1 - i + 0.5,
            winner_genre,
            ha="center",
            va="center",
            fontsize=10,
        )
        groups[np.unravel_index(neuron_index, network_shape)] = genres.index(
            winner_genre
        )
    groups = np.flip(groups, axis=0)

    sns.heatmap(groups, cmap=sns.light_palette("#a275ac", len(genres)), annot=False)

    plot_config.print_plot()
    plt.clf()


def plot_activation_per_genre(
    genre_activations: List[Counter[str]],
    network_config: KohonenConfig,
    genre: Optional[str] = None,
    plot_config: PlotConfig = PlotConfig(),
) -> None:
    K = network_config.grid_size
    R = network_config.neighbours_radius
    LEARNING_RATE = network_config.learning_rate
    MAX_EPOCHS = network_config.epochs

    def neuron_display_name(
        neuron_activations: Counter, genre_name: Optional[str]
    ) -> str:
        return (
            neuron_activations.get(genre_name)
            if genre_name is not None
            else neuron_activations.total()
        )

    plt.figure(figsize=plot_config.fig_size if plot_config.fig_size else (10, 8))
    k = range(K)
    for i, j in product(k, k):
        plt.text(
            j + 0.5,
            K - 1 - i + 0.5,
            neuron_display_name(genre_activations[i * K + j], genre),
            ha="center",
            va="center",
            fontsize=10,
        )
    plt.title(
        ("Activaciones totales" if genre is None else f"Activaciones de {genre}")
        + f". Red {K}x{K} con η(0)={str(LEARNING_RATE)}, R={str(R)} y {MAX_EPOCHS} epochs"
    )

    network_shape = (K, K)
    groups = np.zeros(network_shape, dtype=int)
    for neuron_index, genre_activations_in_neuron in enumerate(genre_activations):
        activations = (
            genre_activations_in_neuron.get(genre, 0)
            if genre is not None
            else genre_activations_in_neuron.total()
        )
        groups[np.unravel_index(neuron_index, network_shape)] += activations
    groups = np.flip(groups, axis=0)

    sns.heatmap(groups, cmap="viridis", annot=False)

    plot_config.print_plot()
    plt.clf()


def plot_activation_matrix(
    winners: np.ndarray[int],
    kohonen_config: KohonenConfig,
    plot_config: PlotConfig = PlotConfig(),
) -> None:
    """Plots the activation matrix for a given variable,
    or for all of them if none is given"""
    K = kohonen_config.grid_size
    R = kohonen_config.neighbours_radius
    LEARNING_RATE = kohonen_config.learning_rate
    MAX_EPOCHS = kohonen_config.epochs

    plt.figure(figsize=plot_config.fig_size if plot_config.fig_size else (10, 8))

    plt.title(
        f"Activación de neuronas. Red {K}x{K} con η(0)={str(LEARNING_RATE)}, R={str(R)} y {MAX_EPOCHS} epochs"
    )
    network_shape = (K, K)
    groups = np.zeros(network_shape, dtype=int)
    for winner in winners:
        groups[np.unravel_index(winner, network_shape)] += 1
    groups = np.flip(groups, axis=0)

    sns.heatmap(groups, cmap="viridis", annot=False)

    plot_config.print_plot()
    plt.clf()


def pivot(
    categorized: pl.DataFrame,
    cluster_order: Optional[Iterable[int]] = None,
    normalize=False,
    index: Optional[str] = None,
    column: Optional[str] = None,
):
    if index is None:
        index = dataset.genres
    if column is None:
        column = "cluster"
    categorized = categorized.pivot(
        index=index,
        columns=column,
        values=dataset.imdb_id,
        aggregate_function=pl.len(),
        sort_columns=True,
    ).fill_null(0)

    def scale_to_sum_1(values):
        values = np.array(values)
        return tuple(values / values.sum() * 100)

    if normalize:
        columns = [col for col in categorized.columns if col != index]
        categorized = categorized.select(
            index,
            values=pl.concat_list(pl.all().exclude(index))
            .map_elements(scale_to_sum_1, return_dtype=pl.List(pl.Float64))
            .list.to_struct(fields=columns.__getitem__),
        ).unnest("values")

    if cluster_order is not None:
        categorized = categorized.select(index, *map(str, cluster_order))
    return categorized


def plot_confusion(
    categorized: pl.DataFrame,
    cluster_order: Optional[Iterable[int]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    actual: Optional[str] = None,
    predicted: Optional[str] = None,
    plot_config: PlotConfig = PlotConfig(),
):
    categorized = pivot(
        categorized, cluster_order, normalize=True, index=actual, column=predicted
    )

    ax = sns.heatmap(
        categorized.to_pandas().set_index(actual),
        annot=True,
        cmap="Blues",
        fmt="0.1f",
        vmin=0,
        vmax=100,
        yticklabels=True,
    )
    for t in ax.texts:
        t.set_text(t.get_text() + "%")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    plot_config.print_plot()
    plt.clf()


def get_column(input: Input, column: int):
    return [input.data[i][column] for i in range(len(input.data))]


def make_general_clustering(
    columns: List[str],
    dataset_type=dataset.DatasetType.NORMALIZED,
    base_path=Path("kohonen"),
    show_plots=False,
    standarization=ZScore(),
) -> None:
    print("Generating clustering for all inputs")
    print("Setting up data...")
    inputs = Input()
    full_dataset = dataset.load_dataset(dataset_type).drop(
        [c for c in dataset.columns if c not in columns]
    )
    inputs.load_dataset(full_dataset)
    inputs.clean_input()
    # Dictionary with the data for each column of the input
    column_data = {c: get_column(inputs, i) for i, c in enumerate(columns)}

    inputs = np.array(
        [
            standarization.standardize(inputs.clear_data[i])
            for i in range(len(inputs.clear_data))
        ]
    )

    input_len = len(inputs)

    config: KohonenConfig = load_config("template.config.json")

    K = config.grid_size
    R = config.neighbours_radius
    LEARNING_RATE = config.learning_rate
    INPUT_SIZE = inputs.shape[1]
    MAX_EPOCHS = config.epochs

    initial_weights = []
    for i in range(K**2):
        initial_weights.extend(inputs[np.random.randint(0, input_len)])

    kohonen = KohonenNetwork(
        K,
        R,
        INPUT_SIZE,
        LEARNING_RATE,
        MAX_EPOCHS,
        initial_input=initial_weights,
        radius_update=ProgressiveReduction(),
        similarity=EuclideanSimilarity(),
    )

    print("Training network...")
    kohonen.train(inputs)
    print("Training complete!")

    winners = np.zeros(input_len, dtype=int)

    def init_group() -> List[List[float]]:
        return [[] for _ in range(K**2)]

    # This variable will contain the list of winner values per neuron, per variable
    neuron_winners_per_column: Dict[str, List[List[Union[float, str]]]] = {
        c: init_group() for c in columns
    }

    print("Grouping winners...")
    for i in range(input_len):
        winners[i] = kohonen.get_winner(inputs[i])  # [0, k**2)
        for neuron_winners in neuron_winners_per_column:
            neuron_winners_per_column[neuron_winners][int(winners[i])].append(
                column_data[neuron_winners][i]
            )

    # A list with all the genre activations for each neuron in the matrix
    genre_groups: List[Counter[str]] = [
        Counter([g for g in genre_group if g in genres])
        for genre_group in neuron_winners_per_column[dataset.genres]
    ]

    plot_activation_matrix(
        winners,
        config,
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"activation_matrix_k_{K}.svg"),
        ),
    )
    plot_activation_per_genre(
        genre_groups,
        config,
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"activation_per_genre_k_{K}.svg"),
        ),
    )
    plot_activation_per_genre(
        genre_groups,
        config,
        "Action",
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"activation_per_genre_action_k_{K}.svg"),
        ),
    )
    plot_activation_per_genre(
        genre_groups,
        config,
        "Comedy",
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"activation_per_genre_comedy_k_{K}.svg"),
        ),
    )
    plot_activation_per_genre(
        genre_groups,
        config,
        "Drama",
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"activation_per_genre_drama_k_{K}.svg"),
        ),
    )
    plot_neuron_classification(
        genre_groups,
        genres,
        config,
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"neuron_classification_k_{K}.svg"),
        ),
    )
    plot_u_matrix(
        kohonen,
        plot_config=PlotConfig(
            show=show_plots, save_file=base_path.joinpath(f"u_matrix_k_{K}.svg")
        ),
    )


def classify_data(
    columns: List[str],
    genres: List[str],
    train_ratio=0.7,
    dataset_type=dataset.DatasetType.NORMALIZED,
    base_path=Path("kohonen"),
    show_plots=False,
    standarization=ZScore(),
) -> None:
    print(f"Generating clustering for genres {', '.join(genres)}")
    print("Setting up data...")
    inputs = Input()
    full_dataset = (
        dataset.load_dataset(dataset_type)
        .drop([c for c in dataset.columns if c not in columns + [dataset.imdb_id]])
        .filter(pl.col(dataset.genres).is_in(genres))
    )
    train, test = dataset.split_dataframe(full_dataset, train_ratio)
    inputs.load_dataset(train.drop(dataset.imdb_id))
    inputs.clean_input()
    # Dictionary with the data for each column of the input
    column_data = {c: get_column(inputs, i) for i, c in enumerate(columns)}

    inputs = np.array(
        [
            standarization.standardize(inputs.clear_data[i])
            for i in range(len(inputs.clear_data))
        ]
    )

    input_len = len(inputs)

    config: KohonenConfig = load_config("template.config.json")

    K = config.grid_size
    R = config.neighbours_radius
    LEARNING_RATE = config.learning_rate
    INPUT_SIZE = inputs.shape[1]
    MAX_EPOCHS = config.epochs

    initial_weights = []
    for i in range(K**2):
        initial_weights.extend(inputs[np.random.randint(0, input_len)])

    kohonen = KohonenNetwork(
        K,
        R,
        INPUT_SIZE,
        LEARNING_RATE,
        MAX_EPOCHS,
        initial_input=initial_weights,
        radius_update=ProgressiveReduction(),
        similarity=EuclideanSimilarity(),
    )

    print("Training network...")
    kohonen.train(inputs)
    print("Training complete!")

    winners = np.zeros(input_len, dtype=int)

    def init_group() -> List[List[float]]:
        return [[] for _ in range(K**2)]

    # This variable will contain the list of winner values per neuron, per variable
    neuron_winners_per_column: Dict[str, List[List[Union[float, str]]]] = {
        c: init_group() for c in columns
    }

    print("Grouping winners...")
    for i in range(input_len):
        winners[i] = kohonen.get_winner(inputs[i])  # [0, k**2)
        for neuron_winners in neuron_winners_per_column:
            neuron_winners_per_column[neuron_winners][int(winners[i])].append(
                column_data[neuron_winners][i]
            )

    # A list with all the genre activations for each neuron in the matrix
    genre_groups: List[Counter[str]] = [
        Counter([g for g in genre_group if g in genres])
        for genre_group in neuron_winners_per_column[dataset.genres]
    ]

    print("Classifying test data...")
    classification_matrix = [
        neuron_activations.most_common(1)[0][0] for neuron_activations in genre_groups
    ]

    parsed_test = (
        test.drop(dataset.imdb_id)
        .with_columns(pl.lit(np.random.random(test.height)).alias(dataset.genres))
        .to_numpy()
        .astype(np.float_)
    )
    predictions = []
    wins = []
    for test_input in parsed_test:
        wins.append(kohonen.get_winner(test_input))
        predictions.append(classification_matrix[kohonen.get_winner(test_input)])
    test = test.with_columns(pl.Series(name="predictions", values=predictions))
    test.write_csv("out/data/predictions.csv", separator=";")
    plot_confusion(
        test.select(dataset.imdb_id, actual=dataset.genres, predicted="predictions"),
        actual="actual",
        predicted="predicted",
        xlabel="predicted",
        ylabel="actual",
        plot_config=PlotConfig(
            show=show_plots,
            save_file=base_path.joinpath(f"confusion_matrix_k_{K}.svg"),
            tight_layout=False,
        ),
    )


if __name__ == "__main__":
    columns = [
        dataset.budget,
        dataset.genres,
        dataset.original_title_len,
        dataset.overview_len,
        dataset.popularity,
        dataset.production_companies,
        dataset.production_countries,
        dataset.revenue,
        dataset.runtime,
        dataset.spoken_languages,
        dataset.vote_average,
        dataset.vote_count,
        dataset.imdb_id,
    ]

    genres = ["Action", "Comedy", "Drama"]

    dataset_type = dataset.DatasetType.NORMALIZED

    make_general_clustering(columns[:-1])

    classify_data(columns[:-1], genres, base_path=Path("kohonen", "classification"))
