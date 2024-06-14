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
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from imdb_api import TMDB_API, OMDB_API, Movie
from tqdm import tqdm

load_dotenv()


def cast_raw_dataset(target: DatasetType = DatasetType.DEFAULT):
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
    default_dataset = fill_null_values(default_dataset)

    print("Nulls filled. Null count")
    print(default_dataset.null_count())
    print(f"Saving dataset as {target.name}. Path: {target.value['path']}")
    save_dataset(default_dataset, target)


def fill_null_values(df: pl.DataFrame) -> pl.DataFrame:
    # Get repeated imdb_ids and remove repeated rows.
    # Then we'll get the api values for those rows as a single source of truth
    repeated_ids = (
        df.join(df.group_by(imdb_id).agg(pl.count().alias("count")), on=imdb_id)
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
        if (
            imdb_id_val not in repeated_ids
            and len([v for v in row.values() if v is None]) == 0
        ):
            continue
        if imdb_id_val is None:
            continue
        movie = TMDB_API().get_movie_info(imdb_id_val)
        for key, value in movie.__dict__.items():
            if row[key] is None:
                rows[idx][key] = value

    filled_df = pl.DataFrame(rows)
    return filled_df


cast_raw_dataset(DatasetType.API_FILLED)
# dataset.filter(pl.any_horizontal(pl.all().is_null())).drop(
#     [overview, imdb_id, original_title]
# ).write_csv(Path("out", "null_values.csv").open("+w"))
