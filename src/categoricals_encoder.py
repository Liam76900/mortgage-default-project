import pandas as pd

def encode_categoricals(loan_default_data, categorical_columns):
    return pd.get_dummies(loan_default_data, columns=categorical_columns, drop_first=True)