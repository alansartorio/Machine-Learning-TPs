from dataset import (
    DatasetType,
    load_dataset,
    save_dataset,
    budget,
    production_companies,
    production_countries,
    revenue,
    runtime,
    spoken_languages,
    vote_count,
    overview,
    imdb_id,
    original_title,
)
import dataset
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from imdb_api import TMDB_API, OMDB_API, Movie
from tqdm import tqdm

load_dotenv()


def cast_raw_dataset():
    raw_dataset = load_dataset(DatasetType.RAW)
    default_dataset = raw_dataset.with_columns(
        pl.col(budget).cast(pl.UInt64, strict=True),
        pl.col(production_companies).cast(pl.UInt16, strict=True),
        pl.col(production_countries).cast(pl.UInt16, strict=True),
        pl.col(revenue).cast(pl.UInt64, strict=True),
        pl.col(runtime).cast(pl.UInt32, strict=True),
        pl.col(spoken_languages).cast(pl.UInt8, strict=True),
        pl.col(vote_count).cast(pl.UInt16, strict=True),
    )
    default_dataset = default_dataset.drop_nulls([imdb_id])   # 45 values out of 5.505

    try:
        raise FileNotFoundError()
        default_dataset = load_dataset(DatasetType.API_FILLED)
    except FileNotFoundError:
        default_dataset = fill_null_values(default_dataset)

        print("Nulls filled. Null count")
        print(default_dataset.null_count())
        save_dataset(default_dataset, DatasetType.API_FILLED)

    default_dataset = convert_to_numerical(default_dataset)
    save_dataset(default_dataset, DatasetType.NUMERICAL)

    default_dataset = normalize_numbers(default_dataset)
    save_dataset(default_dataset, DatasetType.NORMALIZED)


def drop_repeated_values(df: pl.DataFrame):
    repeated_ids = df.group_by(dataset.imdb_id).len("count").filter(pl.col("count") > 1)
    # repeated_lines = df.group_by(pl.all()).len("count").filter(pl.col("count") > 1)

    for id in repeated_ids.get_column(dataset.imdb_id):
        group = df.filter(pl.col(dataset.imdb_id) == id)

        df_without_this = df.filter(pl.col(dataset.imdb_id) != id)
        # Only add the first row with that ID
        df = pl.concat((df_without_this, group.head(1)))

    return df.sort(imdb_id)


def fill_null_values(df: pl.DataFrame) -> pl.DataFrame:
    # Get repeated imdb_ids and remove repeated rows.
    # Then we'll get the api values for those rows as a single source of truth
    repeated_ids = (
        df.join(df.group_by(imdb_id).agg(pl.len().alias("count")), on=imdb_id)
        .filter(pl.col("count") > 1)
        .select(imdb_id)
        .unique()
        .to_series()
        .to_list()
    )
    df = df.unique(imdb_id)
    rows = df.to_dicts()

    for idx, row in tqdm(
        enumerate(rows), total=len(rows), desc="Filling null values", colour="#a970ff"
    ):
        imdb_id_val = row["imdb_id"]
        is_repeated_row = imdb_id_val in repeated_ids
        if (
            not is_repeated_row
            and len([v for v in row.values() if v is None]) == 0
        ):
            continue
        
        movie = TMDB_API().get_movie_info(imdb_id_val)
        for key, value in movie.__dict__.items():
            if is_repeated_row or row[key] is None:
                rows[idx][key] = value

    return pl.DataFrame(rows)


def convert_to_numerical(df: pl.DataFrame) -> pl.DataFrame:
    def str_len(col):
        return pl.col(col).map_elements(len, return_dtype=pl.Int64)
    return df.with_columns(
        str_len(dataset.original_title).alias(dataset.original_title_len),
        str_len(dataset.overview).alias(dataset.overview_len),
    )


def normalize_numbers(df: pl.DataFrame) -> pl.DataFrame:
    columns = df.get_columns()
    numeric_columns = [
        column.name for column in columns if column.dtype in pl.NUMERIC_DTYPES
    ]
    all_columns = [column.name for column in columns]

    mins = df.select(numeric_columns).min().to_numpy()[0, :]
    maxs = df.select(numeric_columns).max().to_numpy()[0, :]
    spreads = maxs - mins

    for column_name, min, spread in zip(numeric_columns, mins, spreads):
        df = df.with_columns((pl.col(column_name) - min) / spread)

    # sort columns to match original df
    df = df.select(all_columns)

    return df


cast_raw_dataset()
