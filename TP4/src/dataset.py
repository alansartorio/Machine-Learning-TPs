import polars as pl
from enum import Enum
from pathlib import Path


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
    # Dataset with values casted and null values filled with the TMDB API
    #   The rows without an imdb_id were dropped (45 rows out of 5.505)
    NULL_FILLED = {
        "path": "null_filled.csv",
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
    # This can be used to define other datasets to load


def load_dataset(type: DatasetType = DatasetType.DEFAULT) -> pl.DataFrame:
    with INPUT_PATH.joinpath(type.value.get("path")) as input_file:
        return pl.read_csv(input_file, separator=";", dtypes=type.value.get("dtypes"))


def save_dataset(dataset: pl.DataFrame, as_type: DatasetType) -> None:
    with INPUT_PATH.joinpath(as_type.value.get("path")) as output_file:
        dataset.write_csv(output_file, separator=";")
