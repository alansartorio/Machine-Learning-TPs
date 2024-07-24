import umap
import numpy as np
import pandas as pd
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict
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
    plot_data,
    PlotConfig,
)


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


if __name__ == "__main__":
    cancer = load_breast_cancer()
    features = cancer.feature_names

    data = np.c_[cancer.data, cancer.target]
    columns = np.append(cancer.feature_names, ["target"])
    data_df = pd.DataFrame(data, columns=columns)

    X = data_df[data_df.columns[:-1]]
    y = data_df.target

    # print(data_df.describe().T)

    # Pre-Analysis
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    algorithms = ["PCA", "KPCA", "LLE", "UMAP", "Supervised UMAP"]

    reductions: Dict[str, Dataset] = {}

    for alg in algorithms:

        match alg:
            case "PCA":
                pca = PCA(n_components=2, random_state=42)
                reduction = Dataset(
                    train=pca.fit_transform(X_train_scaled),
                    test=pca.transform(X_test_scaled),
                )
            case "LLE":
                lle_model = LocallyLinearEmbedding(n_components=2, random_state=42)
                reduction = Dataset(
                    train=lle_model.fit_transform(X_train_scaled),
                    test=lle_model.transform(X_test_scaled),
                )
            case "KPCA":
                kpca_model = KernelPCA(n_components=2, kernel="rbf", random_state=42)
                reduction = Dataset(
                    train=kpca_model.fit_transform(X_train_scaled),
                    test=kpca_model.transform(X_test_scaled),
                )
            case "UMAP":
                UMAP_model = umap.UMAP(
                    n_neighbors=5, min_dist=0.3, n_components=2, random_state=42
                )
                reduction = Dataset(
                    train=UMAP_model.fit_transform(X_train_scaled),
                    test=UMAP_model.transform(X_test_scaled),
                )
            case "Supervised UMAP":
                UMAP_model = umap.UMAP(
                    n_neighbors=5, min_dist=0.3, n_components=2, random_state=42
                )
                reduction = Dataset(
                    train=UMAP_model.fit_transform(X_train_scaled, y=y_train),
                    test=UMAP_model.transform(X_test_scaled),
                )
            case _:
                raise ValueError(f"Unknown algorithm: {alg}")

        reductions.update({alg: reduction})

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
