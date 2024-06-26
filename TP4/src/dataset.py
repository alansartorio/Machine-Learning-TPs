import polars as pl
from enum import Enum
from pathlib import Path
from typing import Tuple
from numpy import random


INPUT_PATH = Path("input")


columns = (
    "budget",
    "genres",
    "imdb_id",
    "original_title",
    "overview",
    "popularity",
    "production_companies",
    "production_countries",
    "release_date",
    "revenue",
    "runtime",
    "spoken_languages",
    "vote_average",
    "vote_count",
)

(
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
) = columns

original_title_len = original_title + "_len"
overview_len = overview + "_len"


DEFAULT_DTYPES = {
    budget: pl.UInt64,
    genres: pl.Utf8,
    imdb_id: pl.Utf8,
    original_title: pl.Utf8,
    overview: pl.Utf8,
    popularity: pl.Float32,
    production_companies: pl.UInt16,
    production_countries: pl.UInt16,
    release_date: pl.Date,
    revenue: pl.UInt64,
    runtime: pl.UInt32,
    spoken_languages: pl.UInt8,
    vote_average: pl.Float32,
    vote_count: pl.UInt16,
}


class DatasetType(Enum):
    # Original dataset
    RAW = {
        "path": "raw.csv",
        "dtypes": {
            budget: pl.Float64,
            genres: pl.Utf8,
            imdb_id: pl.Utf8,
            original_title: pl.Utf8,
            overview: pl.Utf8,
            popularity: pl.Float32,
            production_companies: pl.Float32,
            production_countries: pl.Float32,
            release_date: pl.Date,
            revenue: pl.Float32,
            runtime: pl.Float32,
            spoken_languages: pl.Float32,
            vote_average: pl.Float32,
            vote_count: pl.Float32,
        },
    }
    # Default dataset to use for training the algorithms
    #   Datatypes were casted to the most appropriate ones
    DEFAULT = {
        "path": "default.csv",
        "dtypes": DEFAULT_DTYPES,
    }
    # Dataset with values casted and null values filled with the TMDB API
    #   The rows without an imdb_id were dropped (45 rows out of 5.505)
    NULL_FILLED = {
        "path": "null_filled.csv",
        "dtypes": DEFAULT_DTYPES,
    }
    # Dataset with values casted and null values filled with the TMDB API
    #   The rows without an imdb_id were dropped (45 rows out of 5.505)
    #   The repeated imdb_id values were dropped and fully filled with api data as
    #       a single source of truth
    API_FILLED = {
        "path": "api_filled.csv",
        "dtypes": DEFAULT_DTYPES,
    }
    # NULL_FILLED but with all duplicate rows removed
    UNIQUE_ROWS = {
        "path": "unique_rows.csv",
        "dtypes": {
            budget: pl.UInt64,
            genres: pl.Utf8,
            imdb_id: pl.Utf8,
            original_title: pl.Utf8,
            overview: pl.Utf8,
            popularity: pl.Float32,
            production_companies: pl.UInt16,
            production_countries: pl.UInt16,
            release_date: pl.Date,
            revenue: pl.UInt64,
            runtime: pl.UInt32,
            spoken_languages: pl.UInt8,
            vote_average: pl.Float32,
            vote_count: pl.UInt16,
        },
    }
    # UNIQUE_ROWS but with all string columns converted to numbers
    NUMERICAL = {
        "path": "numerical.csv",
        "dtypes": {
            budget: pl.UInt64,
            genres: pl.Utf8,
            imdb_id: pl.Utf8,
            original_title: pl.Utf8,
            original_title_len: pl.UInt64,
            overview: pl.Utf8,
            overview_len: pl.UInt64,
            popularity: pl.Float32,
            production_companies: pl.UInt16,
            production_countries: pl.UInt16,
            release_date: pl.Date,
            revenue: pl.UInt64,
            runtime: pl.UInt32,
            spoken_languages: pl.UInt8,
            vote_average: pl.Float32,
            vote_count: pl.UInt16,
        },
    }
    # NUMERICAL but with all numeric columns normalized between [0, 1]
    NORMALIZED = {
        "path": "normalized.csv",
        "dtypes": {
            budget: pl.Float64,
            genres: pl.Utf8,
            imdb_id: pl.Utf8,
            original_title: pl.Utf8,
            original_title_len: pl.Float64,
            overview: pl.Utf8,
            overview_len: pl.Float64,
            popularity: pl.Float64,
            production_companies: pl.Float64,
            production_countries: pl.Float64,
            release_date: pl.Date,
            revenue: pl.Float64,
            runtime: pl.Float64,
            spoken_languages: pl.Float64,
            vote_average: pl.Float64,
            vote_count: pl.Float64,
        },
    }
    # This can be used to define other datasets to load


def load_dataset(type: DatasetType = DatasetType.DEFAULT) -> pl.DataFrame:
    with INPUT_PATH.joinpath(type.value.get("path")) as input_file:
        return pl.read_csv(input_file, separator=";", dtypes=type.value.get("dtypes"))


def split_dataframe(
        df: pl.DataFrame, train_ratio=0.7
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:  # (train, test)
        df = df.sample(fraction=1, shuffle=True, with_replacement=True)
        train, test = (
            df.with_columns(random=pl.lit(random.rand(df.height)))
            .with_columns(train=pl.col("random") > train_ratio)
            .sort(pl.col("train"), descending=False)
            .partition_by("train")
        )
        return train.drop("random").drop("train"), test.drop("random").drop("train")

def load_dataset_split(
    type: DatasetType = DatasetType.DEFAULT,
    train_ratio=0.7,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    return split_dataframe(load_dataset(type), train_ratio)


def save_dataset(dataset: pl.DataFrame, as_type: DatasetType) -> None:
    with INPUT_PATH.joinpath(as_type.value.get("path")) as output_file:
        dataset.write_csv(output_file, separator=";")
