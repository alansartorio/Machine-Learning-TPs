import umap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.tree import DecisionTreeClassifier
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
    # plot_class_count(data_df, PlotConfig(show=False, save_file="class_count.svg"))
    # plot_corr_matrix(
    #     data_df, reduced_columns, PlotConfig(show=False, save_file="corr_matrix.svg")
    # )
    # plot_corr_target(
    #     data_df,
    #     "target",
    #     reduced_columns,
    #     PlotConfig(show=False, save_file="corr_target.svg"),
    # )
    plot_distributions(
        data_df, reduced_columns, PlotConfig(show=False, save_file="violin_plots.svg")
    )

    exit()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43
    )

    # Reduce dimensionality
    # Apply StandardScaler to scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_train_PCA = pca.fit_transform(X_train_scaled)
    X_test_PCA = pca.transform(X_test_scaled)

    # LLE
    lle_model = LocallyLinearEmbedding(n_components=2, random_state=42)
    X_train_LLE = lle_model.fit_transform(X_train_scaled)
    X_test_LLE = lle_model.transform(X_test_scaled)

    # Kernel PCA
    kpca_model = KernelPCA(n_components=2, kernel="rbf", random_state=42)
    X_train_KPCA = kpca_model.fit_transform(X_train_scaled)
    X_test_KPCA = kpca_model.transform(X_test_scaled)

    UMAP_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    X_train_UMAP = UMAP_model.fit_transform(X_train_scaled)
    X_test_UMAP = UMAP_model.transform(X_test_scaled)

    UMAP_model_2 = umap.UMAP(
        n_neighbors=5, min_dist=0.3, n_components=2, random_state=42
    )
    X_train_UMAP_supervised = UMAP_model_2.fit_transform(X_train_scaled, y=y_train)
    X_test_UMAP_supervised = UMAP_model_2.transform(X_test_scaled)

    X_train_list = [
        X_train_PCA,
        X_train_KPCA,
        X_train_LLE,
        X_train_UMAP,
        X_train_UMAP_supervised,
    ]
    titles = ["PCA", "KPCA", "LLE", "UMAP", "Supervised UMAP"]
    plot_data(X_train_list, y_train, titles, "dimensionality_reduction_train")

    X_test_list = [
        X_test_PCA,
        X_test_KPCA,
        X_test_LLE,
        X_test_UMAP,
        X_test_UMAP_supervised,
    ]
    titles = ["PCA", "KPCA", "LLE", "UMAP", "Supervised UMAP"]
    plot_data(X_test_list, y_test, titles, "dimensionality_reduction_test")

    evaluate_model(X_train, y_train, X_test, y_test)

    evaluate_and_plot_model(X_train_PCA, y_train, X_test_PCA, y_test)
    evaluate_and_plot_model(X_train_KPCA, y_train, X_test_KPCA, y_test)
    evaluate_and_plot_model(X_train_LLE, y_train, X_test_LLE, y_test)
    evaluate_and_plot_model(X_train_UMAP, y_train, X_test_UMAP, y_test)
    evaluate_and_plot_model(
        X_train_UMAP_supervised, y_train, X_test_UMAP_supervised, y_test
    )
