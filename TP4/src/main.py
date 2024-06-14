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
    default_dataset = default_dataset.drop_nulls([imdb_id]) # 45 values out of 5.505
    # This takes a lot of time to execute
    default_dataset = fill_null_values(default_dataset) 

    print("Nulls filled. Null count")
    print(default_dataset.null_count())
    save_dataset(default_dataset, DatasetType.NULL_FILLED)


def fill_null_values(df: pl.DataFrame) -> pl.DataFrame:
    rows = df.to_dicts()
    
    for idx, row in tqdm(enumerate(rows),total=len(rows),desc="Filling null values"):
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

cast_raw_dataset()

# dataset.filter(pl.any_horizontal(pl.all().is_null())).drop(
#     [overview, imdb_id, original_title]
# ).write_csv(Path("out", "null_values.csv").open("+w"))
