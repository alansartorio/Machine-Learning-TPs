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
    default_dataset = default_dataset.drop_nulls([imdb_id])  # 45 values out of 5.505
    # This takes a lot of time to execute
    try:
        default_dataset = load_dataset(DatasetType.NULL_FILLED)
    except FileNotFoundError:
        default_dataset = fill_null_values(default_dataset)

        print("Nulls filled. Null count")
        print(default_dataset.null_count())
        save_dataset(default_dataset, DatasetType.NULL_FILLED)

    default_dataset = drop_repeated_values(default_dataset)

    print("Unique rows")
    print("Repeated IMDB ids:", len(default_dataset) - default_dataset.n_unique(imdb_id))
    save_dataset(default_dataset, DatasetType.UNIQUE_ROWS)


    default_dataset = convert_to_numerical(default_dataset)

    save_dataset(default_dataset, DatasetType.NUMERICAL)


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
    rows = df.to_dicts()

    for idx, row in tqdm(enumerate(rows), total=len(rows), desc="Filling null values"):
        if len([v for v in row.values() if v is None]) == 0:
            continue
        imdb_id = row["imdb_id"]
        if imdb_id is None:
            continue
        movie = TMDB_API().get_movie_info(imdb_id)
        for key, value in movie.__dict__.items():
            if row[key] is None:
                rows[idx][key] = value

    filled_df = pl.DataFrame(rows)
    return filled_df


def convert_to_numerical(df: pl.DataFrame) -> pl.DataFrame:
    str_len = lambda col: pl.col(col).map_elements(len, return_dtype=pl.Int64)
    df = df.with_columns(
        str_len(dataset.original_title).alias(dataset.original_title_len),
        str_len(dataset.overview).alias(dataset.overview_len),
    )
    return df


cast_raw_dataset()

# dataset.filter(pl.any_horizontal(pl.all().is_null())).drop(
#     [overview, imdb_id, original_title]
# ).write_csv(Path("out", "null_values.csv").open("+w"))
