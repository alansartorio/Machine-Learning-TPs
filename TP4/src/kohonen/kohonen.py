import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt

from config import Input, load_config, KohonenConfig
from radius_update import IdentityUpdate, ProgressiveReduction, RadiusUpdate
from similarity import EuclideanSimilarity, Similarity
from standardization import ZScore

import sys

# setting path
sys.path.append("src")
import dataset


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


if __name__ == "__main__":
    print("Setting up data...")
    inputs = Input()

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
    ]

    dataset_type = dataset.DatasetType.NORMALIZED
    inputs.load_dataset(
        dataset.load_dataset(dataset_type).drop(
            [c for c in dataset.columns if c not in columns]
        )
    )
    inputs.clean_input()

    def get_column(input: Input, column: int):
        return [input.data[i][column] for i in range(len(input.data))]

    column_data = {c: get_column(inputs, i) for i, c in enumerate(columns)}

    standarization = ZScore()

    inputs = np.array(
        [
            standarization.standardize(inputs.clear_data[i])
            for i in range(len(inputs.clear_data))
        ]
    )

    config: KohonenConfig = load_config("template.config.json")

    K = config.grid_size
    R = config.neighbours_radius
    LEARNING_RATE = config.learning_rate
    INPUT_SIZE = inputs.shape[1]
    MAX_EPOCHS = config.epochs

    initial_weights = []
    for i in range(K**2):
        initial_weights.extend(inputs[np.random.randint(0, len(inputs))])

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

    winners = np.zeros(len(inputs))

    def init_group() -> List[List[float]]:
        return [[] for _ in range(K**2)]

    groups = {c: init_group() for c in columns}

    print("Grouping winners...")
    for i in range(len(inputs)):
        winners[i] = kohonen.get_winner(inputs[i])  # [0, k**2)
        for group in groups:
            groups[group][int(winners[i])].append(column_data[group][i])

    for i in range(len(next(iter(groups.values())))):
        for group in groups:
            if group == dataset.genres:
                continue
            groups[group][i] = (
                np.mean(groups[group][i]) if len(groups[group][i]) > 0 else 0
            )

    genre_groups: List[List[str]] = [
        list(set(genre_group)) for genre_group in groups[dataset.genres]
    ]
    groups_dict = {f"Group {i}": g for i, g in enumerate(genre_groups)}

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
                "input_size": INPUT_SIZE,
                "size": len(inputs),
            },
            "groups": groups_dict,
            "weights": kohonen.weights.tolist(),
        }
        json.dump(result, file, ensure_ascii=False, indent=4)

    ### --- HEATMAP CON NOMBRES -----
    matrix = np.zeros((K, K), dtype=int)

    # Process the data and fill the matrix

    for group, countries in groups_dict.items():
        if countries:
            row, col = divmod(int(group.split()[1]), K)
            matrix[K - 1 - row, col] = len(countries)

    # Add group names to each cell
    plt.figure(figsize=(10, 8))
    for i in range(K):
        for j in range(K):
            plt.text(
                j + 0.5,
                K - 1 - i + 0.5,
                "\n".join(groups_dict.get(f"Group {i * K + j}", "")),
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.title(
        f"Groups Heatmap {K}x{K} with η(0)={str(LEARNING_RATE)}, R={str(R)} and {MAX_EPOCHS} epochs"
    )

    groups = np.array(list(map(lambda x: len(x), genre_groups))).reshape((K, K))
    groups = np.flip(groups, axis=0)

    sns.heatmap(groups, cmap="viridis", annot=False)
    plt.show()

    ####
    ### NEIGHBOURS ###

    plt.title("Unified Distance Matrix Heatmap")

    unified_distance = kohonen.get_unified_distance_matrix()
    unified_distance = np.flip(unified_distance, axis=0)

    sns.heatmap(unified_distance, cmap="gray", annot=True)
    plt.show()
