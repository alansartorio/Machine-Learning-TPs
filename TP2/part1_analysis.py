from numpy import multiply
from part1_fetch import (
    creditability,
    age,
    account_balance,
    duration_of_credit,
    most_valuable_available_asset,
    credit_amount,
    purpose,
)
import polars as pl
from polars import DataFrame
from typing import List, Set, Sequence, Optional, Callable, Tuple, Dict, Iterable
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from part1_fetch import get_unbalanced_data, columns as col_names
import textwrap
from dataclasses import dataclass
from copy import copy
from pathlib import Path
from part1_utils import IntRange, VariableBalancer
from enum import Enum
import math

matplotlib.use("Qt5Agg")
plt.switch_backend("Qt5Agg")
sns.set_theme()

type ValueMapping = Dict[str, Dict[int, str]]


class DistType(Enum):
    HIST = 1
    BOXPLOT = 2
    COUNTPLOT = 3
    BARPLOT = 4


@dataclass
class PlotConfig:
    show_fig: bool = True
    save_fig: bool = False
    save_dir: str = "./"
    fig_name: str = "plot.svg"

    def save_and_show(self) -> None:
        if self.show_fig:
            plt.show()
        if self.save_fig:
            if not (p := Path(self.save_dir)).exists():
                p.mkdir()
            plt.tight_layout()
            plt.savefig(self.save_path)

    @property
    def save_path(self) -> Path:
        return Path(self.save_dir, self.fig_name)

    def with_fig_name(self, name: str) -> "PlotConfig":
        c = copy(self)
        c.fig_name = name
        return c


def credit_result_analysis(df: DataFrame):
    ax = sns.countplot(df, x=creditability, hue=creditability)
    ax.set_title("Distribución de las devoluciones del crédito")
    ax.set_xlabel("Creditabilidad")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No devolvió", "Devolvió"])

    plt.tight_layout()
    plt.savefig("./plots/part_1/creditability_distribution.svg")

    plt.clf()


def analyze_and_transform_data(
    df: DataFrame, columns: Sequence[str]
) -> Tuple[DataFrame, ValueMapping]:
    # plot = sns.displot(
    #     data=df,
    #     kind='hist',
    #     x=duration_of_credit,
    # )
    config = PlotConfig(show_fig=True, save_fig=False, save_dir="./plots/part_1")
    variables = set(filter(lambda c: c != creditability, columns))
    value_mapping = {
        column.name: {value: str(value) for value in column.unique()}
        for column in df.get_columns()
    }

    ##########
    # Categorical variables -> bar charts for distributions, box plots for outliers
    # Continuous variables -> histograms
    ##########

    # overview_of_distribution(df, variables, config.with_fig_name('distribution_overview.svg'))
    df, _, balancers = balance_variables(df, variables, value_mapping)
    age_balancer = balancers[age]
    plot_distribution_for_variable(
        df,
        age,
        config,
        xticks=age_balancer.get_balanced_values(),
        xtick_labels=[
            age_balancer.balanced_value_to_str(x)
            for x in age_balancer.get_balanced_values()
        ],
    )

    # plot_distribution_for_variable(df, credit_amount, config, 'box')

    return df, value_mapping


def adjust_column(df: DataFrame, column: str, adjust) -> DataFrame:
    df.get_column(column).apply(adjust)


def plot_distribution_for_variable(
    df: DataFrame,
    variable: str,
    config: PlotConfig,
    after: Optional[Callable] = None,
    type: DistType = DistType.HIST,
    xticks: Optional[Iterable] = None,
    xtick_labels: Optional[Iterable[str]] = None,
    **kwargs,
):
    column = df.get_column(variable)
    match type:
        case DistType.HIST:
            ax = sns.histplot(
                df, x=variable, kde=False, hue=creditability, multiple="stack", **kwargs
            )
        case DistType.BOXPLOT:
            ax = sns.boxplot(column, **kwargs)
        case DistType.COUNTPLOT:
            ax = sns.histplot(
                df,
                x=variable,
                kde=False,
                hue=creditability,
                multiple="stack",
                discrete=True,
                **kwargs,
            )
        case DistType.BARPLOT:
            ax = sns.lineplot(df, x=variable, hue=creditability, **kwargs)
        case _:
            raise Exception(
                f"Type {type} not recognized. Accepted values are {[e.name for e in DistType]}"
            )
    if after is not None:
        after(ax)
    ax.set_title(variable)
    if xticks is not None:
        ax.set_xticks(xticks)
        if xtick_labels is not None:
            ax.set_xticklabels(xtick_labels, rotation=45)
    config.save_and_show()
    plt.clf()


def overview_of_distribution(
    df: DataFrame, columns: Sequence[str], config: PlotConfig
) -> None:
    n_cols = 6
    n_vars = len(columns)
    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(n_cols * 5, n_rows * 4)
    )
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    ax_flat = axes.flatten()

    for i, col in enumerate(columns):
        ax: plt.Axes = ax_flat[i]
        sns.histplot(df, x=col, ax=ax, kde=False, hue=creditability, multiple="stack")
        ax.set_title(textwrap.fill(col, width=25))
        # ax.set_axis_off()
        # ax.grid(visible=True)

    for ax in ax_flat[i + 1 :]:
        ax.set_visible(False)
    for ax in ax_flat:
        for xlabel_i in ax.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        for xlabel_i in ax.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
        for tick in ax.get_xticklines():
            tick.set_visible(False)
        for tick in ax.get_yticklines():
            tick.set_visible(False)

    config.save_and_show()

    plt.clf()


def balance_variables(
    df: DataFrame, variables: Sequence[str], value_mappings: ValueMapping
) -> Tuple[DataFrame, ValueMapping, Dict[str, VariableBalancer]]:
    age_balancer = VariableBalancer(
        [
            IntRange(0, 27),
            IntRange(27, 33),
            IntRange(33, 42),
        ],
        IntRange(42, float("inf")),
    )

    balancers = {age: age_balancer}

    for variable in variables:
        if variable in balancers.keys():
            balancer = balancers[variable]
            value_mappings[variable] = balancer.get_mappings()
            series: pl.Series = df.get_column(variable).map_elements(
                balancer.balance, pl.Int64
            )
            df = df.with_columns(series.alias(variable))
    return df, value_mappings, balancers


def do_thing(df, column):
    return (
        df.group_by([pl.col(column), pl.col(creditability)])
        .count()
        .pivot(index=column, columns=creditability, values="count")
        .fill_null(0)
        .with_columns((pl.col("1") / (pl.col("1") + pl.col("0"))).alias("1"))
        .melt(
            id_vars=column,
            value_vars=["1"],
            variable_name=creditability,
            value_name="percentage",
        )
    )
    # .with_columns((1-pl.col('1')).alias('0')) \


if __name__ == "__main__":
    df = get_unbalanced_data()
    credit_result_analysis(df)
    config = PlotConfig(show_fig=False, save_fig=True, save_dir="./plots/part_1")
    variables = set(filter(lambda c: c != creditability, col_names))

    age_df = do_thing(df, age)
    plot_distribution_for_variable(
        age_df,
        age,
        config.with_fig_name("creditability_percentage_by_age.svg"),
        type=DistType.BARPLOT,
        y="percentage",
        after=lambda ax: ax.set_ylim((0, 1)),
    )
    plot_distribution_for_variable(
        df,
        age,
        config.with_fig_name("age_distribution.svg"),
    )
    credit_df = do_thing(
        df.with_columns((pl.col(credit_amount) // 500) * 500), credit_amount
    )
    plot_distribution_for_variable(
        credit_df,
        credit_amount,
        config.with_fig_name("creditability_percentage_by_credit_amount.svg"),
        type=DistType.BARPLOT,
        y="percentage",
        after=lambda ax: ax.set_ylim((0, 1)),
    )
    plot_distribution_for_variable(
        df,
        credit_amount,
        config.with_fig_name("credit_distribution.svg"),
    )
    plot_distribution_for_variable(
        df,
        account_balance,
        config.with_fig_name("account_balance_distribution.svg"),
        type=DistType.COUNTPLOT,
    )
    plot_distribution_for_variable(
        df,
        most_valuable_available_asset,
        config.with_fig_name("most_valuable_asset_distribution.svg"),
        type=DistType.COUNTPLOT,
    )
    plot_distribution_for_variable(
        df,
        duration_of_credit,
        config.with_fig_name("duration_of_credit_distribution.svg"),
    )

    sns.set_theme(font_scale=1)
    overview_of_distribution(
        df, variables, config.with_fig_name("distribution_overview.svg")
    )
