from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from json import JSONDecoder
from pathlib import Path
import numpy as np
from typing import Union
from polars import DataFrame

import sys
# setting path
sys.path.append('src')
from dataset import load_dataset, DatasetType



@dataclass
class ConfigData(ABC):
    @staticmethod
    @abstractmethod
    def from_json(json_file: dict) -> "ConfigData":
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass


@dataclass
class KohonenConfig(ConfigData):
    neighbours_radius: int = 1
    epochs: int = 500
    grid_size: int = 5
    learning_rate: float = 0.1

    @staticmethod
    def from_json(json_file: dict) -> "KohonenConfig":
        config_data = KohonenConfig()
        config_data.neighbours_radius = (
            json_file.get("neighbours_radius") or config_data.neighbours_radius
        )
        config_data.epochs = json_file.get("epochs") or config_data.epochs
        config_data.grid_size = json_file.get("grid_size") or config_data.grid_size
        config_data.learning_rate = (
            json_file.get("learning_rate") or config_data.learning_rate
        )
        return config_data

    def to_json(self) -> dict:
        return {
            "neighbours_radius": self.neighbours_radius,
            "epochs": self.epochs,
            "grid_size": self.grid_size,
            "learning_rate": self.learning_rate,
        }

def load_config(filename: str = "config.json") -> ConfigData:
    """
    Load the configs data from the configs file
    """
    path = Path("src", "kohonen", filename)
    if not path.exists():
        raise Exception(f"The selected config file does not exist: {path}")

    with open(path, "r+", encoding="utf-8") as config_f:
        return JSONDecoder(object_hook=KohonenConfig.from_json).decode(config_f.read())


class Input:
    data: np.ndarray
    clear_data: np.ndarray

    def __init__(self):
        self.data = np.zeros(1)
        self.clear_data = np.zeros(1)

    def load_dataset(self, dataset: Union[DatasetType,DataFrame]):
        dt = load_dataset(dataset) if type(dataset) == DatasetType else dataset
        values_list = dt.to_numpy()
        self.data = np.array(values_list)
        self.clear_data = np.array(values_list)

    def clean_input(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if isinstance(self.data[i][j], str):
                    self.clear_data[i][j] = self.string_to_number(self, self.data[i][j])
        self.clear_data = self.clear_data.astype(np.float_)

    @staticmethod
    def string_to_number(self, string: np.ndarray):
        number = 0
        for i in range(len(string)):
            number += ord(string[i])
        return int(number)
    
    def __str__(self):
        return "\n".join(map(str, self.data))


if __name__ == "__main__":
    input = Input()
    input.load_dataset(DatasetType.DEFAULT)

    print(input)
