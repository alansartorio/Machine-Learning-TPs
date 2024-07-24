import umap
import numpy as np
import pandas as pd
import re
import unicodedata
from dataclasses import dataclass
import json
from typing import List, Dict, Any
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from analyze import (
    plot_class_count,
    plot_corr_matrix,
    plot_corr_target,
    plot_distributions,
    evaluate_and_plot_model,
    evaluate_model,
    plot_umap_comparison,
    plot_data,
    evaluate_umap,
    PlotConfig,
)

config: Dict[str, Any] = json.load(open("config.json"))


@dataclass
class Dataset:
    train: np.ndarray
    test: np.ndarray


def slug(input_string):
    normalized_string = (
        unicodedata.normalize("NFKD", input_string)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    lower_case_string = normalized_string.lower()

    slug = re.sub(r"[^a-z0-9]+", "_", lower_case_string)
    slug = slug.strip("_")

    return slug


def default_config(filename: str, figsize=(7, 5)) -> PlotConfig:
    return PlotConfig(
        show=False,
        save_file=filename,
        tight_layout=True,
        fig_size=figsize,
    )

def plot_pre_analysis(dat_df: pd.DataFrame):
    reduced_columns = list(filter(lambda c: c.startswith("mean"), data_df.columns))
    plot_class_count(
        data_df,
        PlotConfig(
            show=False, save_file="class_count.svg", tight_layout=True, fig_size=(8, 6)
        ),
    )
    plot_corr_matrix(
        data_df,
        reduced_columns,
        PlotConfig(
            show=False, save_file="corr_matrix.svg", tight_layout=True, fig_size=(10, 9)
        ),
    )
    plot_corr_target(
        data_df,
        "target",
        reduced_columns,
        PlotConfig(
            show=False, save_file="corr_target.svg", tight_layout=True, fig_size=(10, 9)
        ),
    )
    plot_distributions(
        data_df,
        reduced_columns,
        PlotConfig(
            show=False,
            save_file="violin_plots.svg",
            tight_layout=True,
            fig_size=(15, 10),  # (20, 30)
        ),
    )

def compute_embeddings(
        X_train: np.ndarray,
        X_test: np.ndarray,
        algorithms: List[str],
) -> Dict[str, Dataset]: 
    """
    Generate embeddings for the selected algorithms
    """
    seed = config.get("Seed")
    reductions: Dict[str, Dataset] = {}

    for alg in algorithms:

        match alg:
            case "PCA":
                pca = PCA(n_components=2, random_state=seed)
                reduction = Dataset(
                    train=pca.fit_transform(X_train),
                    test=pca.transform(X_test),
                )
            case "LLE":
                lle_model = LocallyLinearEmbedding(n_components=2, random_state=seed)
                reduction = Dataset(
                    train=lle_model.fit_transform(X_train),
                    test=lle_model.transform(X_test),
                )
            case "KPCA":
                kernel = config.get("KPCA", {}).get("Kernel", "rbf")
                kpca_model = KernelPCA(n_components=2, kernel=kernel, random_state=seed)
                reduction = Dataset(
                    train=kpca_model.fit_transform(X_train),
                    test=kpca_model.transform(X_test),
                )
            case "UMAP":
                umap_conf = config.get("UMAP", {})

                UMAP_model = umap.UMAP(
                    n_neighbors=umap_conf.get("n_neighbors", 5),
                    min_dist=umap_conf.get("min_dist", 0.3), 
                    n_components=2, 
                    metric=umap_conf.get("metric", "euclidean"),
                    init=umap_conf.get("init", "spectral"),
                    n_epochs=umap_conf.get("n_epochs", None),
                    learning_rate=umap_conf.get("learning_rate", 1.0),
                    random_state=seed
                )
                reduction = Dataset(
                    train=UMAP_model.fit_transform(X_train),
                    test=UMAP_model.transform(X_test),
                )
            case "Supervised UMAP":
                umap_conf = config.get("Supervised UMAP", {})

                UMAP_model = umap.UMAP(
                    n_neighbors=umap_conf.get("n_neighbors", 5),
                    min_dist=umap_conf.get("min_dist", 0.3), 
                    n_components=2, 
                    metric=umap_conf.get("metric", "euclidean"),
                    init=umap_conf.get("init", "spectral"),
                    n_epochs=umap_conf.get("n_epochs", None),
                    learning_rate=umap_conf.get("learning_rate", 1.0),
                    random_state=seed
                )
                reduction = Dataset(
                    train=UMAP_model.fit_transform(X_train, y=y_train),
                    test=UMAP_model.transform(X_test),
                )
            case _:
                raise ValueError(f"Unknown algorithm: {alg}")

        reductions.update({alg: reduction})
    return reductions

def test_umap(
        X_train, 
        X_test,
        n_neighbors_values=[15],
        min_dist_values=[0.1],
        ) -> List[Dict[str, Any]]:
    """
    Generate umap embeddings for different values of n_neighbors and min_dist
    """
    n_components = 2
    umap_conf = config.get("UMAP", {})
    results = []

    for n_neighbors, min_dist in itertools.product(n_neighbors_values, min_dist_values):
        print("Testing UMAP with n_neighbors =", n_neighbors, "and min_dist =", min_dist)
        UMAP_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=umap_conf.get("metric", "euclidean"),
            init=umap_conf.get("init", "spectral"),
            n_epochs=umap_conf.get("n_epochs", None),
            learning_rate=umap_conf.get("learning_rate", 1.0),
        )
        results.append({
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "dataset": Dataset(
                train=UMAP_model.fit_transform(X_train),
                test=UMAP_model.transform(X_test),
            )
        })

    return results


if __name__ == "__main__":
    cancer = load_breast_cancer()
    features = cancer.feature_names

    data = np.c_[cancer.data, cancer.target]
    columns = np.append(cancer.feature_names, ["target"])
    data_df = pd.DataFrame(data, columns=columns)

    X = data_df[data_df.columns[:-1]]
    y = data_df.target

    # plot_pre_analysis(data_df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get("Seed")
    )

    evaluate_and_plot_model(
        X_train,
        y_train,
        X_test,
        y_test,
        "original",
        default_config("confusion_matrix_original.svg"),
    )

    # Reduce dimensionality
    # Apply StandardScaler to scale features
    scaler = StandardScaler()

    # Test umap with different values for n_neighbors and min_dist
    X_scaled = scaler.fit_transform(X)

    n_neighbors_values=[5, 30, 50, 100]
    min_dist_values=[0,0.1,0.5,1]

    print("Testing dataset for different UMAP configurations")
    umap_results = test_umap(
        X_scaled, 
        X_scaled,
        n_neighbors_values=n_neighbors_values,
        min_dist_values=min_dist_values,
    )
    
    plot_umap_comparison(
        n_neighbors_values,
        min_dist_values,
        [r["dataset"].train for r in umap_results],
        y,
        PlotConfig(
            show=False,
            save_file="umap_comparison",
            tight_layout=True,
            fig_size=(20, 20),
        )
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test model performance for different UMAP configurations
    print("Testing model performance for different UMAP configurations")
    umap_results = test_umap(
        X_train_scaled, 
        X_test_scaled,
        n_neighbors_values=n_neighbors_values,
        min_dist_values=min_dist_values,
    )
    evaluate_umap(
        umap_results,
        n_neighbors_values,
        min_dist_values,
        y_train,
        y_test,
        default_config("umap_evaluation.svg"),
    )

    algorithms = ["PCA", "KPCA", "UMAP", "Supervised UMAP"]

    reductions = compute_embeddings(X_train_scaled, X_test_scaled, algorithms)    

    X_train_list = list(map(lambda x: x.train, reductions.values()))
    X_test_list = list(map(lambda x: x.test, reductions.values()))

    plot_data(
        X_train_list,
        y_train,
        algorithms,
        PlotConfig(
            show=False,
            save_file="dimensionality_reduction_train",
            tight_layout=True,
            fig_size=(20, 20),
        ),
    )

    plot_data(
        X_test_list,
        y_test,
        algorithms,
        PlotConfig(
            show=False,
            save_file="dimensionality_reduction_test",
            tight_layout=True,
            fig_size=(20, 20),
        ),
    )

    for alg, dataset in reductions.items():
        evaluate_and_plot_model(
            dataset.train,
            y_train,
            dataset.test,
            y_test,
            alg,
            default_config(f"confusion_matrix_{slug(alg)}.svg"),
            default_config(f"decision_boundary_{slug(alg)}.svg"),
        )
