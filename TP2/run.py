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

credit_category = (df.get_column(credit_amount) / 1000).round().cast(pl.Int64)
df = df.with_columns(credit_category.alias(credit_amount))
df = df.with_columns(df.get_column(creditability).cast(pl.String))

print(df)

with open("tree.dot", 'w') as graph_file:
    print(repr(tp2.train(df, creditability)), file=graph_file)
