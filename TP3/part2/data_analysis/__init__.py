from PIL import Image
from pathlib import Path
from polars import DataFrame, Series, UInt8, Int8
import polars as pl
from typing import Callable, Optional, List, Any, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np 

def load_image(filepath: Path, get_class: Callable[[int,int,int,int,int],UInt8]) -> DataFrame:
    """
    Generates a DataFrame from an image file
        Parameters:
            filepath (Path): the file's path
            get_class ((x, y, r, g, b) -> UInt8): a function that determines the class of a pixel given its coordinates and value: (x)
    """
    image = Image.open(filepath)
    width, height = image.size
    pixels = image.load()

    r_values = []
    g_values = []
    b_values = []
    class_values = []
    for y in range(height):
        for x in range(width):
            r,g,b = pixels[x,y]
            r_values.append(r)
            g_values.append(g)
            b_values.append(b)
            class_values.append(get_class(x,y,r,g,b))

    return DataFrame([
        Series('r', r_values, UInt8),
        Series('g', g_values, UInt8),
        Series('b', b_values, UInt8),
        Series('class', class_values, Int8)
    ])

MAX_POINTS=3000

def plot_rgb_projections(df: DataFrame, title: str, show:bool=True, save_path:Optional[Path]=None):
    data = df.sample(MAX_POINTS).drop(['class']).to_numpy()

    data_normalized = data / 255.0

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Red-Green projection
    axs[0].scatter(data_normalized[:, 0], data_normalized[:, 1], c=data_normalized, s=100)
    axs[0].set_xlabel('Rojo')
    axs[0].set_ylabel('Verde')
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].set_title('Proyección Rojo-Verde')

    # Green-Blue projection
    axs[1].scatter(data_normalized[:, 1], data_normalized[:, 2], c=data_normalized, s=100)
    axs[1].set_xlabel('Verde')
    axs[1].set_ylabel('Blue')
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_title('Proyección Verde-Blue')

    # Blue-Red projection
    axs[2].scatter(data_normalized[:, 2], data_normalized[:, 0], c=data_normalized, s=100)
    axs[2].set_xlabel('Azul')
    axs[2].set_ylabel('Rojo')
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].set_title('Proyección Azul-Rojo')

    fig.suptitle(title, fontsize=16)
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    plt.clf()

def plot_rgb_cube(df: DataFrame, title: str, show:bool=True, save_path:Optional[Path]=None):
    ax = plt.axes(projection='3d')

    data = df.sample(MAX_POINTS).drop(['class']).to_numpy()

    data_normalized = data / 255.0

    ax.scatter(
        data_normalized[:, 0], 
        data_normalized[:, 1], 
        data_normalized[:, 2], 
        c=data_normalized, 
        s=100
        )

    ax.set_xlabel('Rojo')
    ax.set_ylabel('Verde')
    ax.set_zlabel('Azul')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.clf()

def plot_confusion_matrix(confusion_matrix: NDArray[Any], labels, title: str, show:bool=True, save_path:Optional[Path]=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    plt.clf()

def plot_metrics(
        metrics: List[float], 
        metric_names: List[str], 
        class_names: List[str], 
        title: str, 
        show:bool=True, 
        save_path:Optional[Path]=None
    ):
    x = np.arange(len(class_names))
    width = 0.2 

    fig, ax = plt.subplots()
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, metric, width, label=metric_names[i])

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(class_names)
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    plt.clf()

def paint_image(
        image_path: Path, 
        classifications: NDArray, 
        color_map: Dict[int, List[int]],
        output_path: Path=Path('./image.jpg'),
        show=True
    ):
    image_pixels = np.array(Image.open(image_path))
    classifications = classifications.reshape((image_pixels.shape[0], image_pixels.shape[1]))
    for i in range(image_pixels.shape[0]):
        for j in range(image_pixels.shape[1]):
            class_label = classifications[i, j]
            image_pixels[i, j] = color_map[class_label]
    modified_image = Image.fromarray(image_pixels)
    modified_image.save(output_path)
    if show:
        modified_image.show()