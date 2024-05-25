import numpy as np

class SingleData:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return f'{self.inputs} -> {self.outputs}'
