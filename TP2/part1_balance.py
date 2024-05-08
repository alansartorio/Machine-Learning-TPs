import polars as pl
from polars import DataFrame

from part1_fetch import get_unbalanced_data, creditability


df = get_unbalanced_data()

counts = df.group_by(creditability).len()

print(counts)
min_count: int = counts.select('len').to_series().min()

print(min_count)

sample_0 = df \
    .filter(pl.col(creditability).eq(0)) \
    .sample(n=min_count, with_replacement=False)


sample_1 = df \
    .filter(pl.col(creditability).eq(1)) \
    .sample(n=min_count, with_replacement=False)

pl.concat([sample_0, sample_1]).write_csv('out/german_credit.csv')
