import sys
sys.path.append('src')

from data_loader import load_data

loan_default_data = load_data('data/Loan_Default.csv')

print(loan_default_data.shape)
print(loan_default_data.columns)
print(loan_default_data.head())
print(loan_default_data.dtypes)
print(loan_default_data.isnull().sum())