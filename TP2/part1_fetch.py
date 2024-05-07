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

def get_data() -> pl.DataFrame:
    return pl.read_csv("input/german_credit.csv", new_columns=columns)
