from .data_analysis import load_image, plot_rgb_cube, plot_rgb_projections, plot_confusion_matrix, paint_image
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, Literal
from polars import DataFrame
import polars as pl
import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC 
from time import time
from dataclasses import dataclass, asdict
import json
from itertools import product


data_frames: Dict[str, DataFrame] = dict(
    cow=None,   
    vaca=None,
    pasto=None,
    cielo=None,
    train=None,
    test=None,
)

config: Dict[str,Any] = dict(
    show_plot=False,
    input_path=Path('./input/'),
    output_path=Path('./out/part2'),
    plots_path=Path('./plots/part2')
)

CLASSES = {
    'None': 3,
    'vaca': -1,
    'pasto': 0,
    'cielo': 1,
}

# # Old classes
# CLASSES = {
    # 'None': 0,
    # 'vaca': 1,
    # 'pasto': 2,
    # 'cielo': 3, 
# }

@dataclass
class PolyParams:
    degree:int=3
    gamma:Union[Literal['scale','auto'],float]='scale'
    coef0: float=0

@dataclass
class RBFParams:
    gamma:Union[Literal['scale','auto'],float]='scale'

@dataclass
class SigmoidParams:
    gamma:Union[Literal['scale','auto'],float]='scale'
    coef0:float=0


def input_file(filename: str) -> Path:
    return config['input_path'].joinpath(filename)

def output_file(filename: str) -> Path:
    return config['output_path'].joinpath(filename)

def plot_file(filename: str) -> Path:
    return config['plots_path'].joinpath(filename)

def load_datasets():
    with_class = lambda c: (lambda x,y,r,g,b: CLASSES[c])
    
    data_frames['cow'] = load_image(input_file('cow.jpg'), with_class('None'))
    for name in ['vaca', 'cielo', 'pasto']:
        data_frames[name] = load_image(input_file(f'{name}.jpg'), with_class(name))

def print_sizes():
    print("Size for vaca.jpg:", f"{data_frames["vaca"].height:,}")
    print("Size for pasto.jpg:", f"{data_frames["pasto"].height:,}")
    print("Size for cielo.jpg:", f"{data_frames["cielo"].height:,}")

def print_class_sizes(df: DataFrame):
    print("Vaca:", f"{df.filter(pl.col('class') == CLASSES['vaca']).select(pl.len()).item():,}")
    print("Cielo:", f"{df.filter(pl.col('class') == CLASSES['cielo']).select(pl.len()).item():,}")
    print("Pasto:", f"{df.filter(pl.col('class') == CLASSES['pasto']).select(pl.len()).item():,}")

def generate_rgb_cubes():
    plot_rgb_cube(data_frames['cow'], 'cow.jpg', show=config['show_plot'], save_path=plot_file('cow_rgb.svg'))
    plot_rgb_projections(data_frames['cow'], 'Proyecciones cow.jpg', show=config['show_plot'], save_path=plot_file('cow_rgb_projections.svg'))
    plot_rgb_cube(data_frames['vaca'], 'vaca.jpg', show=config['show_plot'], save_path=plot_file('vaca_rgb.svg'))
    plot_rgb_projections(data_frames['vaca'], 'Proyecciones vaca.jpg', show=config['show_plot'], save_path=plot_file('vaca_rgb_projections.svg'))
    plot_rgb_cube(data_frames['pasto'], 'pasto.jpg', show=config['show_plot'], save_path=plot_file('pasto_rgb.svg'))
    plot_rgb_projections(data_frames['pasto'], 'Proyecciones pasto.jpg', show=config['show_plot'], save_path=plot_file('pasto_rgb_projections.svg'))
    plot_rgb_cube(data_frames['cielo'], 'cielo.jpg', show=config['show_plot'], save_path=plot_file('cielo_rgb.svg'))
    plot_rgb_projections(data_frames['cielo'], 'Proyecciones cielo.jpg', show=config['show_plot'], save_path=plot_file('cielo_rgb_projections.svg'))

def check_datasets():
    df: DataFrame = data_frames['cow']
    print("Null Count", df.null_count())

def adapt_datasets():
    data_frames['vaca'] = data_frames['vaca'].sample(n=35_000, shuffle=True)
    data_frames['pasto'] = data_frames['pasto'].sample(n=35_000, shuffle=True)
    data_frames['cielo'] = data_frames['cielo'].sample(n=35_000, shuffle=True)

def generate_train_test(train_ratio=0.7):
    def split_dataframe(df: DataFrame, train_ratio=0.7) -> Tuple[DataFrame, DataFrame]: # (train, test)
        df = df.sample(fraction=1, shuffle=True, with_replacement=True)
        train, test = df.with_columns(random = pl.lit(np.random.rand(df.height))).with_columns(train = pl.col('random') > train_ratio).sort(pl.col('train'), descending=False).partition_by('train')
        return train.drop('random').drop('train'), test.drop('random').drop('train')
    vaca_sets = split_dataframe(data_frames['vaca'],train_ratio)
    cielo_sets = split_dataframe(data_frames['cielo'],train_ratio)
    pasto_sets = split_dataframe(data_frames['pasto'],train_ratio)

    train = pl.concat([vaca_sets[0], cielo_sets[0], pasto_sets[0]])    
    test = pl.concat([vaca_sets[1], cielo_sets[1], pasto_sets[1]])    
    data_frames['train'] = train
    data_frames['test'] = test
    return train, test

def train_model(C=1.0,kernel='linear',kernel_params=None) -> Tuple[SVC, float]:
    """Train the SVM model with the given parameters
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty. For an intuitive visualization of the effects
        of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm. If
        none is given, 'rbf' will be used. If a callable is given it is used to
        pre-compute the kernel matrix from data matrices; that matrix should be
        an array of shape ``(n_samples, n_samples)``. For an intuitive
        visualization of different kernel types see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_kernels.py`.
    """
    x_train = data_frames['train'].drop('class')
    y_train = data_frames['train'].get_column('class')
    extra_params = asdict(kernel_params) if kernel_params is not None else {}
    model = SVC(
        C=C,
        kernel=kernel,
        verbose=False,
        tol=0.001, # Tolerance for error stop
        max_iter=-1, # Unlimited
        decision_function_shape='ovo', # one-versus-one or one-versus-rest
        cache_size=1000,
        **extra_params
    )

    print(f"Training for kernel {kernel} and C={C}...")
    start_time = time()
    model.fit(x_train, y_train)
    end_time = time()
    elapsed = end_time-start_time
    print(f"Model trained in {elapsed} seconds")
    return model, elapsed

def test_model(model: SVC) -> Tuple[List, List]:
    x_test = data_frames['test'].drop('class')
    y_test = data_frames['test'].get_column('class') 

    y_pred = model.predict(x_test)
    return y_test.to_list(), y_pred.tolist()

def build_confusion_matrix(y_true, y_pred) -> NDArray[Any]:
    unique_labels = np.unique(y_true)
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

    return matrix

def calculate_metrics(confusion_matrix: NDArray[Any]) -> Tuple[List[float],List[float],List[float],float]:
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (fp + fn + tp)

    precision: np.ndarray = tp / (tp + fp)
    recall: np.ndarray = tp / (tp + fn)
    f1_score: np.ndarray = 2 * (precision * recall) / (precision + recall)
    accuracy: float = tp.sum() / confusion_matrix.sum()

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)

    return precision.tolist(), recall.tolist(), f1_score.tolist(), accuracy

def get_class_labels(y: List[int]) -> List[str]:
    def find_label(n: int) -> str:
        for k,i in CLASSES.items():
            if i == n:
                return k
    return [find_label(c) for c in np.unique(y)]

@dataclass
class ModelResult:
    kernel: str
    C:float
    y_true: List
    y_pred: List
    confusion_matrix: np.ndarray
    train_seconds: float
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    accuracy: float

def save_result(path: Path, result: ModelResult):
    with path.open('+w') as file:
        json.dump(dict(
            kernel=result.kernel,
            C=result.C,
            y_true=result.y_true,
            y_pred=result.y_pred,
            confusion_matrix=result.confusion_matrix.tolist(),
            train_seconds=result.train_seconds,
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1_score,
            accuracy=result.accuracy,
        ), file)

def load_result(filepath: Path) -> ModelResult:
    with filepath.open('r') as file:
        return ModelResult(**json.load(file))

def test_values(kernel_values, c_values,kernel_params=None):
    for kernel, C in product(kernel_values, c_values):
        model, elapsed_time = train_model(C,kernel,kernel_params)
        y_true, y_pred = test_model(model)

        matrix = build_confusion_matrix(y_true, y_pred)

        c_str = str(C).replace('.','_')
        plot_confusion_matrix(
            matrix, 
            get_class_labels(y_true), 
            f"Matriz de Confusión. Kernel={kernel} | C={C:.2f}", 
            False, 
            plot_file(f"confusion_matrix_{kernel}_{c_str}.svg")
        )

        precision, recall, f1_score, accuracy = calculate_metrics(matrix)
        print(f"""Results for model:
    - precision: {precision}
    - recall: {recall}
    - f1-score: {f1_score}
    - accuracy: {accuracy}        
            """)
        result = ModelResult(
            kernel, 
            C, 
            y_true, 
            y_pred, 
            matrix, 
            elapsed_time, 
            precision, 
            recall, 
            f1_score, 
            accuracy
        )
        save_result(output_file(f"result_{kernel}_{c_str}.json"), result)

def classify_image(model: SVC, image_path: Path, output_path: Path, show_result=True):
    COLORS = {
        'red':[255,0,0],
        'green':[0,255,0],
        'blue':[0,0,255]
    }

    x_values = load_image(image_path, lambda x,y,r,g,b:0).drop('class')
    print(f"Classifying image {image_path}...")
    pixel_pred = model.predict(x_values)
    paint_image(
        image_path,
        pixel_pred,
        {
            CLASSES['vaca']:COLORS['red'],
            CLASSES['cielo']:COLORS['blue'],
            CLASSES['pasto']:COLORS['green']
        },
        output_path,
        show_result
    )

load_datasets()
adapt_datasets()
print_sizes()
generate_rgb_cubes()
train, test = generate_train_test(0.7)

print("Train", train)
print("Train split classes:")
print_class_sizes(train)
print("Test", test)
print("Test split classes:")
print_class_sizes(test)

KERNELS = ['linear', 'poly','rbf']
C_VALUES = [1.0, 0.75, 0.5, 0.25, 0.1]

test_values(KERNELS, C_VALUES)
test_values(['sigmoid'], [0.1])

# # Test with specific parameters for kernel
# for gamma in ['scale', 0.001, 0.005, 0.01]:
#     print(f"Training for gamma {gamma}")
#     test_values(['rbf'], [1],RBFParams(gamma))

# # Rank results by accuracy
# print("70% training")
# results_dir = './out/part2/split_0_7'
# accuracies = [{'value': load_result(f).accuracy, 'file': f.name} for f in Path(results_dir).iterdir()]
# accuracies.sort(reverse=True,key=lambda i: i['value'])
# for a in accuracies:
#     print(a['file'],f"{a['value']:.3%}")

# print("80% training")
# results_dir = './out/part2/split_0_8'
# accuracies = [{'value': load_result(f).accuracy, 'file': f.name} for f in Path(results_dir).iterdir()]
# accuracies.sort(reverse=True,key=lambda i: i['value'])
# for a in accuracies:
#     print(a['file'],f"{a['value']:.3%}")

# # Print all results
# for f in  Path('./out/part2/split_0_7').iterdir():
#     data = load_result(f)
#     print(f"Labels: {get_class_labels(data.y_true)}")
#     print(f"""Results for model. Kernel={data.kernel}, C={data.C}:
# - precision: {data.precision}
# - recall: {data.recall}
# - f1-score: {data.f1_score}
# - accuracy: {data.accuracy}        
#         """)

# Classify images
BEST_KERNEL = 'rbf'
BEST_C = 1.0

model, _ = train_model(BEST_C,BEST_KERNEL,RBFParams(gamma='scale'))

classify_image(model,input_file('cow.jpg'),plot_file('classified_cow.jpg'),False)
classify_image(model,input_file('vacas_1.jpg'),plot_file('classified_vacas_1.jpg'),False)
classify_image(model,input_file('vacas_2.jpg'),plot_file('classified_vacas_2.jpg'),False)
classify_image(model,input_file('vacas_3.jpg'),plot_file('classified_vacas_3.jpg'),False)


from part1.activation_functions import step_func
from part1.network import Network
from part1.single_data import SingleData
data = [
        SingleData(np.array([r, g, b]), np.array([clazz]))
        for r, g, b, clazz in train.with_columns(pl.when(pl.col("class") == 0).then(1).when(pl.col("class") == 1).then(1).otherwise(-1).alias("class")).iter_rows()
    ]

model = Network.with_random_weights(3, (1,), step_func)

model.train(0.01, data)

x_values = load_image(input_file('cow.jpg'), lambda x,y,r,g,b:0).drop('class')
y_pred = model.evaluate([np.array([r, g, b]) for r,g,b in x_values.iter_rows()])
COLORS = {
    'red':[255,0,0],
    'green':[0,255,0],
    'blue':[0,0,255]
}
paint_image(
    input_file('cow.jpg'),
    y_pred,
    {
        CLASSES['vaca']:COLORS['red'],
        CLASSES['cielo']:COLORS['blue'],
        CLASSES['pasto']:COLORS['green']
    },
    plot_file('manual_classification.jpg'),
    True
)


