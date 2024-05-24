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

line_angle=math.pi/4
margin=0.5

df = create_dataset(margin=margin,line_angle=line_angle,points_per_class=100)
plot_dataset(df, line_angle, margin)