import tp2
import polars as pl

columns = (
    "Creditability",
    "Account Balance",
    "Duration of Credit",
    "Payment Status of Previous Credit",
    "Purpose",
    "Credit Amount",
    "Value Savings/Stocks",
    "Length of current employment",
    "Instalment per cent",
    "Sex & Marital Status",
    "Guarantors",
    "Duration in Current address",
    "Most valuable available asset",
    "Age",
    "Concurrent Credits",
    "Type of apartment",
    "No of Credits at this Bank",
    "Occupation",
    "No of dependents",
    "Telephone",
    "Foreign Worker"
)
creditability, \
    account_balance, \
    duration_of_credit, \
    payment_status, \
    purpose, \
    credit_amount, \
    value_savings, \
    length_of_current_employment, \
    instalment_per_cent, \
    sex_and_marital_status, \
    guarantors, \
    duration_in_current_address, \
    most_valuable_available_asset, \
    age, \
    concurrent_credits, \
    type_of_apartment, \
    no_of_credits_at_this_bank, \
    occupation, \
    no_of_dependents, \
    telephone, \
    foreign_worker = columns
df = pl.read_csv("input/german_credit.csv", new_columns=columns)

import json
def print_unique_ns(df):
    unique_values = {column.name: column.n_unique() for column in df.get_columns()}
    print(json.dumps(unique_values, indent=4))

print_unique_ns(df)

value_mapping = {column.name: {value:str(value) for value in column.unique()} for column in df.get_columns()}

def reduce_column(df, column, factor):
    series = (df.get_column(column) / factor).round().cast(pl.Int64)
    value_mapping[column] = {reduced: f'[{reduced * factor}, {(reduced + 1) * factor})' for reduced in series.unique()}
    return df.with_columns(series.alias(column))

df = reduce_column(df, credit_amount, 2000)
df = reduce_column(df, age, 5)
df = reduce_column(df, duration_of_credit, 10)
df = df.with_columns(df.get_column(creditability).cast(pl.String))

print_unique_ns(df)
# exit()

print(df)

with open("tree.dot", 'w') as graph_file:
    print(tp2.train(df, creditability, value_mapping).to_graphviz(), file=graph_file)
