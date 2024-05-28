import itertools
from itertools import chain, combinations, product
from random import sample, seed, random
from polars import DataFrame, Series
import polars as pl
import numpy as np
from typing import Tuple, List, TypeVar, Optional, Generic, Union
import seaborn as sns
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass

from part1.network import Network
from part1.single_data import SingleData


sns.set_theme()


def random_range(start: float, end: float) -> float:
    if end < start:
        raise ValueError("End cannot be greater than start")
    return start + random() * (end - start)


T = TypeVar("T")


class Collection(Generic[T]):
    def __init__(self) -> None:
        self.data: List[T] = []
        self.size = 0
        super().__init__()

    def add(self, item: T) -> None:
        self.data.append(item)
        self.size += 1

    def pop(self) -> Optional[T]:
        if self.is_empty():
            return None
        self.size -= 1
        return self.data.pop()

    def is_empty(self) -> bool:
        return self.size == 0

    def __iter__(self):
        return self.data

    def __next__(self):
        return next(self.data)

    def __len__(self) -> int:
        return self.size


def sign(n: float) -> int:
    return 1 if n >= 0 else -1


def point_line_distance(a, b, c, x, y) -> float:
    return abs(a * x + b * y + c) / math.sqrt(a * a + b * b)


def point_line_vertical_diff(a, b, c, x, y) -> float:
    return y - x * (-a / b) - c / b


def create_dataset(
    rand_seed=None,
    line_angle: float = 0,
    margin: float = 1,
    points_per_class=100,
    space_size: Tuple[float, float] = (5, 5),
) -> DataFrame:
    if rand_seed is not None:
        seed(rand_seed)

    class1_points: Collection[Tuple[float, float]] = Collection()
    class2_points: Collection[Tuple[float, float]] = Collection()
    x_values = []
    y_values = []
    class_values = []
    while (
        class1_points.size < points_per_class and class2_points.size < points_per_class
    ):
        normal = np.array([-math.sin(line_angle), math.cos(line_angle)])

        x = random_range(-space_size[0], space_size[0])
        y = random_range(-space_size[1], space_size[1])

        distance = point_line_distance(math.tan(line_angle), -1, 0, x, y)

        if distance < margin:
            continue

        class_ = sign(point_line_vertical_diff(math.tan(line_angle), -1, 0, x, y))
        if class_ == 1:
            class1_points.add((x, y))
        else:
            class2_points.add((x, y))

        x_values.append(x)
        y_values.append(y)
        class_values.append(class_)

    return DataFrame(
        [
            Series("x", x_values, pl.Float32),
            Series("y", y_values, pl.Float32),
            Series("class", class_values, pl.Int8),
        ]
    )


def add_bad_points(
    rand_seed=None,
    line_angle: float = 0,
    margin: float = 1,
    points_per_class=100,
    space_size: Tuple[float, float] = (5, 5),
) -> DataFrame:
    if rand_seed is not None:
        seed(rand_seed)

    class1_points: Collection[Tuple[float, float]] = Collection()
    class2_points: Collection[Tuple[float, float]] = Collection()
    x_values = []
    y_values = []
    class_values = []
    while (
        class1_points.size < points_per_class and class2_points.size < points_per_class
    ):
        normal = np.array([-math.sin(line_angle), math.cos(line_angle)])

        x = random_range(-space_size[0], space_size[0])
        y = random_range(-space_size[1], space_size[1])

        distance = point_line_distance(math.tan(line_angle), -1, 0, x, y)

        if distance > margin:
            continue

        class_ = -sign(point_line_vertical_diff(math.tan(line_angle), -1, 0, x, y))
        if class_ == 1:
            class1_points.add((x, y))
        else:
            class2_points.add((x, y))

        x_values.append(x)
        y_values.append(y)
        class_values.append(class_)

    return DataFrame(
        [
            Series("x", x_values, pl.Float32),
            Series("y", y_values, pl.Float32),
            Series("class", class_values, pl.Int8),
        ]
    )


def plot_dataset(
    df: DataFrame,
    line: Union[float, tuple[float, float, float]],
    margin: Optional[float],
    filename: str,
    space_size: Tuple[float, float] = (5, 5),
) -> None:
    ax = sns.scatterplot(df, x="x", y="y", hue="class", palette="RdBu")
    ax.set_xbound(-space_size[0], space_size[0])
    ax.set_ybound(-space_size[1], space_size[1])

    if type(line) is float:
        slope = math.tan(line)
        y = 0
    elif type(line) is tuple:
        a, b, c = line
        slope = -a / b
        y = -c / b
    else:
        raise Exception(type(line))
    line_angle = math.atan(slope)
    ax.axline((0, y), slope=slope)  # Hiperplane
    if margin is not None:
        # Margin
        ax.axline(
            (-math.sin(line_angle) * margin, margin * math.cos(line_angle) + y),
            slope=slope,
            color="#A9A9A9",
            linestyle="dashed",
        )
        ax.axline(
            (math.sin(line_angle) * margin, -margin * math.cos(line_angle) + y),
            slope=slope,
            color="#A9A9A9",
            linestyle="dashed",
        )
    plt.savefig(filename)
    plt.show()


# print(df)


class SVM2d:
    def __init__(self, df: DataFrame, name: str, C: float, k: float = 1.0):
        self.df = df
        self.C = C
        self.ws = np.zeros(len(df.get_columns()) - 1)
        self.b = 0
        self.k = k

        self.name = name

        self.r = 1

        self.percentage_of_data = 0.5

        self.max_iter = 10000

    def train(self):
        iteration = 0

        prev_ws = self.ws
        prev_b = self.b

        while iteration <= self.max_iter:
            # decresing k
            learning_rate = self.k * np.exp(-iteration / self.max_iter)

            data_batch = self.df.sample(
                fraction=self.percentage_of_data, seed=iteration
            )

            for row in data_batch.to_numpy():
                x = np.array(row[:-1])
                y = row[-1]

                if y * (np.dot(self.ws, x) + self.b) < 1:
                    self.ws = self.ws - learning_rate * (self.ws - self.C * y * x)
                    self.b = self.b + learning_rate * self.C * y

                else:
                    self.ws = self.ws - learning_rate * self.ws

            if iteration % 1000 == 0:
                print(
                    f"Iter {iteration} - w: {self.ws}, b: {self.b}, k: {learning_rate}"
                )

            if np.linalg.norm(self.ws - prev_ws) < 1e-4 and abs(self.b - prev_b) < 1e-4:
                break

            iteration += 1

        self.r = 1 / np.linalg.norm(self.ws)
        print(f"w: {self.ws}, b: {self.b}, r: {self.r}, k: {learning_rate}")

    def plot(self):
        hyperplane = np.array([-self.ws[1], self.ws[0]])

        line_angle = math.atan2(hyperplane[1], hyperplane[0])
        plot_dataset(self.df, line_angle, self.r, f"plots/{self.name}.svm.svg")

    def execute(self):
        test = SVM2d(self.df, self.name, C=self.C, k=self.k)
        test.train()
        test.plot()

    def __repr__(self) -> str:
        return f"SimpleSVM(margin={self.margin}, C={self.C}, bs={self.bs})"


line_angle = 13.131
margin = 0.5
df = create_dataset(margin=margin, line_angle=line_angle, points_per_class=100)
bad_points = add_bad_points(margin=margin, line_angle=line_angle, points_per_class=10)
df_bad = pl.concat((df, bad_points))
print(df)
# print(*df.iter_rows(), sep='\n')

plot_dataset(df, line_angle, margin, "plots/tp3-1.svg")


def ej1(df, animation_file, error_file):
    from part1.single_data import SingleData

    data = [
        SingleData(np.array([x, y]), np.array([clazz]))
        for x, y, clazz in df.iter_rows()
    ]

    from .activation_functions import step_func
    from . import network
    from .plot_line import Plot
    from matplotlib.animation import PillowWriter

    model = network.Network.with_random_weights(2, (1,), step_func)
    # ej1_xor_data = (SingleData(np.array([-1, 1]), np.array([1])),
    # SingleData(np.array([-1, -1]), np.array([-1])),
    # SingleData(np.array([1, -1]), np.array([1])),
    # SingleData(np.array([1, 1]), np.array([-1])))

    # data = ej1_xor_data

    minimum_error = None
    minimum_model = None

    plot = Plot([d.inputs for d in data], [d.outputs for d in data], model)

    writer = PillowWriter(fps=10)
    writer.setup(plot.fig, animation_file, dpi=200)

    errors = []
    # print(model.layers[0].weights.flatten())
    # print(model.error(ej1_data))
    try:
        error = model.error(data)
        i = 0
        max_iter = 1000
        while error > 0 and i < max_iter:
            learning_rate = 0.01 * np.exp(-i / max_iter)
            chosen_data = sample(data, 1)[0]
            model.train_single(learning_rate, chosen_data)
            print(model.layers[0].weights.flatten(), model.error(data))
            # print(model.layers[0].weights.flatten(), model.error(ej1_data))
            # for single_data in data:
            # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
            # print()
            if i % 5 == 0:
                plot.update(f"Iteration N°{i}")
                writer.grab_frame()

            error = model.error(data)
            errors.append(error)

            if minimum_error is None or error < minimum_error:
                minimum_error = error
                minimum_model = model.copy()
            i += 1
        plot.update(f"Iteration N°{i}")
    except KeyboardInterrupt:
        pass
    for _ in range(10):
        writer.grab_frame()
    # single_neuron = Network.with_random_weights(1, (2, 3), step_func)
    print("DONE, PRESS 'q'")
    writer.finish()

    plt.show()
    
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(errors)), errors)
    ax.set_ylim(ymin=0)

    ax.set_ylabel("Error")
    ax.set_xlabel("Iteration")
    fig.savefig(error_file)
    plt.show()

    if minimum_model is None or minimum_error is None:
        raise Exception

    print("Best error:", minimum_error, "weights", minimum_model.layers[0].weights)
    c, a, b = minimum_model.layers[0].weights[0]
    c = -c

    return data, minimum_model, a, b, c, minimum_error

def get_margin(data, a, b, c) -> Optional[float]:
    for point in data:
        x, y = point.inputs
        # A point was wrongly classified
        if sign(point_line_vertical_diff(a, b, c, x, y)) != point.outputs[0]:
            return None

    return min(map(lambda p: point_line_distance(a, b, c, *p.inputs), data))


def post_processing(data, a, b, c):
    points = sorted(data, key=lambda p: point_line_distance(a, b, c, *p.inputs))
    partitions = {-1: [], 1: []}
    for point in points:
        # class_ = sign(point_line_vertical_diff(a, b, c, *point.inputs))
        partitions[point.outputs[0]].append(point)

    negative_candidates = partitions[-1][:5]
    positive_candidates = partitions[1][:5]

    print(negative_candidates, positive_candidates)

    def get_middle_line(a, b, point) -> tuple[float, float, float]:
        ab = b - a
        ap = point - a

        # Find new line
        a = a + ap / 2
        b = a + ab

        m = (b[1] - a[1]) / (b[0] - a[0])
        b = a[1] - m * a[0]
        return m, -1, b

    def find_best_combination(negs: list[SingleData], poss: list[SingleData]):
        max_margin = None
        max_margin_line = None
        for (a, b), point in chain(
            product(combinations(negs, 2), poss),
            product(combinations(poss, 2), negs),
        ):
            line = get_middle_line(a.inputs, b.inputs, point.inputs)
            margin = get_margin(data, *line)
            if max_margin is None or (margin is not None and margin > max_margin):
                max_margin = margin
                max_margin_line = line

        if max_margin_line is None or max_margin is None:
            raise Exception
        return max_margin_line, max_margin

    margin_line, margin = find_best_combination(
        negative_candidates, positive_candidates
    )

    return *margin_line, margin


data, model, a, b, c, error = ej1(df, "plots/ej1.a.gif", "plots/ej1.a.error.svg")
print("Error: ", error)

margin = get_margin(data, a, b, c)
if margin is None:
    raise Exception()
plot_dataset(df, (a, b, c), margin, "plots/ej1.a.svg")

a, b, c, margin = post_processing(data, a, b, c)

plot_dataset(df, (a, b, c), margin, "plots/post_processed.svg")

print(a, b, c)

plot_dataset(df_bad, line_angle, margin, "plots/tp3-2.svg")

data, model, a, b, c, error = ej1(df_bad, "plots/ej1.c.gif", "plots/ej1.c.error.svg")
plot_dataset(df_bad, (a, b, c), None, "plots/ej1.c.svg")

# svm = SVM2d(df, "ej1.a", C=0.1, k=0.01).execute()
# svm = SVM2d(df_bad, "ej1.c", C=0.1, k=0.01).execute()
