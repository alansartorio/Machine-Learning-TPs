from random import seed, random
from polars import DataFrame, Series
import polars as pl
import numpy as np
from typing import Tuple, List, TypeVar, Optional, Generic
import seaborn as sns
import matplotlib.pyplot as plt
import math 
from dataclasses import dataclass


sns.set_theme()

def random_range(start: float, end: float) -> float:
    if end < start:
        raise ValueError('End cannot be greater than start')
    return start + random() * (end-start)

T = TypeVar('T')

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

def create_dataset(
        rand_seed=None,
        line_angle:float=0,
        margin:float=1,
        points_per_class=100,
        space_size:Tuple[float, float]=(5,5)
        ) -> DataFrame:
    if rand_seed is not None:
        seed(rand_seed)
    
    class1_points: Collection[Tuple[float,float]] = Collection()
    class2_points: Collection[Tuple[float,float]] = Collection()
    x_values = []
    y_values = []
    class_values = []
    while class1_points.size < points_per_class and class2_points.size < points_per_class:
        normal = np.array([-math.sin(line_angle), math.cos(line_angle)])

        x = random_range(-space_size[0],space_size[0])
        y = random_range(-space_size[1],space_size[1])
        
        point = np.array([x,y])
        proj = point.dot(normal)
        if abs(proj) < margin:
            continue

        class_ = sign(proj)
        if class_ == 1:
            class1_points.add((x,y))
        else:
            class2_points.add((x,y))

        x_values.append(x)
        y_values.append(y)
        class_values.append(class_)

    return DataFrame([Series('x',x_values,pl.Float32),Series('y',y_values,pl.Float32),Series('class',class_values,pl.Int8)])

def plot_dataset(df: DataFrame, line_angle: int, margin: float, space_size:Tuple[float,float]=(5,5)) -> None:
    ax = sns.scatterplot(df, x='x', y='y', hue='class')
    ax.set_xbound(-space_size[0],space_size[0])
    ax.set_ybound(-space_size[1],space_size[1])

    slope = math.tan(line_angle)
    ax.axline((0,0), slope=slope) # Hiperplane
    # Margin
    ax.axline((-math.sin(line_angle)*margin,margin*math.cos(line_angle)), slope=slope, color='#A9A9A9', linestyle='dashed')
    ax.axline((math.sin(line_angle)*margin,-margin*math.cos(line_angle)), slope=slope, color='#A9A9A9', linestyle='dashed')
    plt.show()


# print(df)


class SVM2d:

    def __init__(self, df: DataFrame, margin: float, C: float , k: float = 1.0):
        self.df = df
        self.margin = margin
        self.C = C
        self.ws = np.zeros(len(df.get_columns()) - 1)
        self.b = 0
        self.k = k

        self.r = 1

        self.max_iter = 10000

    def train(self):

        iteration = 0
        has_wrong_class = True
        while iteration <= self.max_iter and has_wrong_class:


            has_wrong_class = False
            # decresing k
            learning_rate = self.k * np.exp(-iteration / self.max_iter)

            data_batch = self.df.sample(fraction=1)


            for row in data_batch.to_numpy():

                x = np.array(row[:-1])
                y = row[-1]

                # print(f'x: {x}, y: {y}, ws: {self.ws}, b: {self.b}')

                if y * (np.dot(self.ws, x) + self.b) < 1:
                    self.ws = self.ws - learning_rate * (self.ws - self.C * y * x)
                    self.b = self.b + learning_rate * self.C * y

                    has_wrong_class = True

                else :
                    self.ws = self.ws - learning_rate * self.ws


            if iteration % 100 == 0:
                print(f'Iter {iteration} - w: {self.ws}, b: {self.b}, k: {learning_rate}')
        
            iteration += 1

        self.r = 1 / np.linalg.norm(self.ws)
        print(f'w: {self.ws}, b: {self.b}, r: {self.r}, k: {learning_rate}')


    def plot(self):
        line_angle = math.atan(self.ws[1])
        plot_dataset(self.df, line_angle, self.r)


    def execute(self, df: DataFrame, margin: float, C: float, k: float = 1.0):
        test = SVM2d(df, margin=margin, C=C, k=k)
        test.train()
        test.plot()

    def __repr__(self) -> str:
        return f'SimpleSVM(margin={self.margin}, C={self.C}, bs={self.bs})'

line_angle=math.pi/4
margin=0.5
df = create_dataset(margin=margin,line_angle=line_angle,points_per_class=100)
print(df)

# plot_dataset(df, line_angle, margin)

from part1.single_data import SingleData

# exit(1)
data = [SingleData(np.array([x, y]), np.array([clazz])) for x, y, clazz in df.iter_rows()]

from .activation_functions import step_func
from . import network
from .plot_line import Plot
import numpy as np
from matplotlib.animation import PillowWriter

model = network.Network.with_random_weights(2, (1,), step_func)
# ej1_xor_data = (SingleData(np.array([-1, 1]), np.array([1])),
                # SingleData(np.array([-1, -1]), np.array([-1])),
                # SingleData(np.array([1, -1]), np.array([1])),
                # SingleData(np.array([1, 1]), np.array([-1])))

# data = ej1_xor_data

plot = Plot([d.inputs for d in data], [d.outputs for d in data], model)

writer = PillowWriter(fps = 10)
writer.setup(plot.fig, 'plots/ej1.gif', dpi=200)

# print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
try:
    while model.error(data) > 0:
        model.train(0.00001, data)
        print(model.layers[0].weights.flatten(), model.error(data))
        # print(model.layers[0].weights.flatten(), model.error(ej1_data))
        # for single_data in data:
            # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
        # print()
        plot.update()
        writer.grab_frame()
except KeyboardInterrupt:
    pass
for _ in range(10):
    writer.grab_frame()
# single_neuron = Network.with_random_weights(1, (2, 3), step_func)
writer.finish()

plt.show()

# print(single_neuron.evaluate(np.array([1])))

