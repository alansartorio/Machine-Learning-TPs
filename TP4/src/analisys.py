from dataset import (
    load_dataset,
    DatasetType,
    budget,
    genres,
    imdb_id,
    original_title,
    overview,
    popularity,
    production_companies,
    production_countries,
    release_date,
    revenue,
    runtime,
    spoken_languages,
    vote_average,
    vote_count,
)
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, Literal, Union, Tuple, List
from pathlib import Path

sns.set_theme()

CATEGORICAL_COLS = [genres]
TEXT_COLS = [overview, original_title, release_date, imdb_id]


@dataclass
class PlotConfig:
    show: bool = True
    output_path: Path = Path("out")
    save_file: Optional[str] = None
    tight_layout: bool = False
    fig_size: Optional[Tuple[int, int]] = None

    def print_plot(self) -> None:
        if self.tight_layout:
            plt.tight_layout()
        if self.save_file is not None:
            plt.savefig(self.output_path.joinpath(self.save_file))
        if self.show:
            plt.show()


dataset = load_dataset(DatasetType.API_FILLED)


def get_numerical_columns(df: pl.DataFrame) -> List[str]:
    return (
        df.to_pandas()
        .select_dtypes(
            include=[
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "uint16",
                "uint32",
                "uint64",
                "uint8",
            ]
        )
        .columns
    )


def plot_two_variables(
    df: pl.DataFrame,
    x: str,
    y: str,
    type: Union[
        Literal["scatterplot"],
        Literal["lineplot"],
        Literal["barplot"],
        Literal["boxplot"],
    ] = "scatterplot",
    title: Optional[str] = None,
    config=PlotConfig(),
) -> None:
    plt.figure(figsize=config.fig_size)
    match type:
        case "scatterplot":
            sns.scatterplot(data=df, x=x, y=y)
        case "lineplot":
            sns.lineplot(data=df, x=x, y=y)
        case "barplot":
            ax = sns.barplot(data=df, x=x, y=y)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
        case "boxplot":
            ax = sns.boxplot(data=df, x=x, y=y)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
        case _:
            raise ValueError(f"Invalid plot type: {type}")
    if title is not None:
        plt.title(title)
    config.print_plot()
    plt.clf()


def plot_categorical_distribution(
    df: pl.DataFrame,
    var: str,
    type: Union[Literal["box"], Literal["violin"]] = "box",
    title: Optional[str] = None,
    config=PlotConfig(),
) -> None:
    plt.figure(figsize=config.fig_size)
    match type:
        case "box":
            print(f"Generating box plot for {var}...")
            sns.boxplot(data=df, x=genres, y=var)
        case "violin":
            print(f"Generating violin plot for {var}...")
            sns.violinplot(data=df, x=genres, y=var)
        case _:
            raise ValueError(f"Invalid plot type: {type}")
    if title is not None:
        plt.title(title)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.subplots_adjust(bottom=0.20)
    config.print_plot()
    plt.clf()
    plt.close()


def covariance_matrix(
    df: pl.DataFrame, title: Optional[str] = None, config=PlotConfig()
) -> None:
    print("Generating covariance matrix...")
    # Adjust plot size so it is not so crowded
    plt.figure(figsize=config.fig_size)  # figsize=(10,8)
    ax = sns.heatmap(df.corr(), annot=True)
    ax.set_xticklabels(labels=df.columns, rotation=30, horizontalalignment="right")
    ax.set_yticklabels(labels=df.columns, rotation=0, horizontalalignment="right")
    # add padding below so the words fit inside the graph
    plt.subplots_adjust(bottom=0.25, left=0.25)
    if title is not None:
        plt.title(title)
    config.print_plot()
    plt.clf()


def pairplot(
    df: pl.DataFrame,
    vars: Optional[List[str]] = None,
    title: Optional[str] = None,
    config=PlotConfig(),
):
    print("Generating pairplot...")
    g = (
        sns.pairplot(df.to_pandas(), vars=vars)
        if vars is not None
        else sns.pairplot(df.to_pandas())
    )
    if title is not None:
        g.figure.suptitle(title, y=1.03)
    config.print_plot()
    plt.clf()


def histogram(
    df: pl.DataFrame, var: str, title: Optional[str] = None, config=PlotConfig()
) -> None:
    g = sns.histplot(data=df, x=var, kde=True)
    if title is not None:
        g.fig.suptitle(title)
    config.print_plot()
    plt.clf()


def countplot(
    df: pl.DataFrame, var: str, title: Optional[str] = None, config=PlotConfig()
) -> None:
    print(f"Computing distribution of {var}...")
    plt.figure(figsize=config.fig_size)
    ax = sns.countplot(data=df, x=var)
    ax.xaxis.set_ticklabels(
        ax.xaxis.get_ticklabels(), rotation=45, horizontalalignment="right"
    )
    plt.subplots_adjust(bottom=0.25)
    if title is not None:
        plt.title(title)
    config.print_plot()
    plt.clf()


def full_histogram(df: pl.DataFrame, title: Optional[str] = None, config=PlotConfig()):
    """Plot histograms for all numerical columns in the DataFrame"""
    print("Computing distribution of numerical variables...")
    df_pandas = df.to_pandas()
    numerical_columns = get_numerical_columns(df)
    df_pandas[numerical_columns].hist(bins=30, figsize=config.fig_size)
    if title is not None:
        plt.suptitle(title)
    plt.subplots_adjust(hspace=0.7)
    config.print_plot()
    plt.clf()


if __name__ == "__main__":
    ### Plot Distribution of features
    ## Numerical features
    full_histogram(
        dataset,
        "Distribución de variables numéricas",
        PlotConfig(
            show=False, save_file="num_hist.svg", tight_layout=False, fig_size=(11, 8)
        ),
    )
    ## Categorical features
    countplot(
        dataset,
        genres,
        "Distribución de géneros",
        PlotConfig(
            show=False, save_file="genres_dist.svg", tight_layout=False, fig_size=(9, 6)
        ),
    )
    ## Plot relationships between variables
    covariance_matrix(
        dataset.drop(CATEGORICAL_COLS + TEXT_COLS),
        "Matriz de correlación de las variables numéricas",
        PlotConfig(show=False, save_file="covariance_matrix.svg", fig_size=(10,8), tight_layout=False),
    )
    pairplot(
        dataset.drop(CATEGORICAL_COLS + TEXT_COLS),
        None,
        "Relación entre variables numéricas",
        PlotConfig(show=False, save_file="pairplot.png", tight_layout=True),
    )
    pairplot(
        dataset.drop(CATEGORICAL_COLS + TEXT_COLS),
        [runtime, popularity, vote_average, vote_count, budget],
        "Relación entre variables numéricas",
        PlotConfig(show=False, save_file="reduced_pairplot.png", tight_layout=True),
    )

    ## Plot distribution of nummerical features acroes the genres category
    # Box Plots
    print("Generating box plots for genres...")
    numerical_columns = get_numerical_columns(dataset)
    for col in numerical_columns:
        plot_categorical_distribution(
            dataset,
            col,
            "box",
            f"Box Plot of {col} by Category",
            PlotConfig(
                show=False,
                fig_size=(12, 6),
                tight_layout=True,
                output_path=Path("out/boxplots"),
                save_file=f"box_{col}.svg",
            ),
        )

        plot_categorical_distribution(
            dataset,
            col,
            "violin",
            f"Box Plot of {col} by Category",
            PlotConfig(
                show=False,
                fig_size=(17, 6),
                tight_layout=True,
                output_path=Path("out/boxplots"),
                save_file=f"box_{col}.svg",
            ),
        )


    # histogram(
    #     dataset.with_columns(
    #         pl.col(overview)
    #         .map_elements(lambda s: len(s.split()), int)
    #         .alias("overview_wordcount")
    #     ),
    #     "overview_wordcount",
    #     "Cantidad de palabras en la descripción de las películas",
    # )
