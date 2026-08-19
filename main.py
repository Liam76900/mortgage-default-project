import sys
sys.path.append('src')

from data_loader import load_data
from age_data_cleaner import encode_age
from missing_values_cleaner import clean_missing_values


loan_default_data = load_data('data/Loan_Default.csv')
loan_default_data = encode_age(loan_default_data)
loan_default_data = clean_missing_values(loan_default_data)

print(loan_default_data.isnull().sum())
print(loan_default_data['Status'].value_counts())
print(loan_default_data[['age', 'age_numeric']].head(10))