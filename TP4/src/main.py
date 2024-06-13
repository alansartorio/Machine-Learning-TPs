from dataset import DatasetType, load_dataset, save_dataset, budget, production_companies, production_countries, revenue, runtime, spoken_languages, vote_count, overview, imdb_id, original_title
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def cast_raw_dataset():
    raw_dataset = load_dataset(DatasetType.RAW)
    default_dataset = raw_dataset.with_columns(
        pl.col(budget).cast(pl.UInt64,strict=True),
        pl.col(production_companies).cast(pl.UInt16,strict=True),
        pl.col(production_countries).cast(pl.UInt16,strict=True),
        pl.col(revenue).cast(pl.UInt64,strict=True),
        pl.col(runtime).cast(pl.UInt32,strict=True),
        pl.col(spoken_languages).cast(pl.UInt8,strict=True),
        pl.col(vote_count).cast(pl.UInt16,strict=True),
    )

    save_dataset(default_dataset,DatasetType.DEFAULT)


dataset = load_dataset()

dataset.filter(
    pl.any_horizontal(pl.all().is_null())
    ).drop(
        [overview,imdb_id,original_title]
    ).write_csv(Path('out','null_values.csv').open('+w'))
