import sys
sys.path.append('src')

from data_loader import load_data
from age_data_cleaner import encode_age
from missing_values_cleaner import clean_missing_values
from income_to_loan_ratio import add_income_to_loan_ratio
from categoricals_encoder import encode_categoricals
from data_splitter import split_data
from train_logistic_regression import train_logistic_regression
from evaluate_model import evaluate_model
from feature_scaler import scale_features

loan_default_data = load_data('data/Loan_Default.csv')
loan_default_data = encode_age(loan_default_data)
loan_default_data = clean_missing_values(loan_default_data)
loan_default_data = add_income_to_loan_ratio(loan_default_data)

categorical_columns = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'Credit_Worthiness', 'open_credit', 'business_or_commercial',
    'Neg_ammortization', 'interest_only', 'lump_sum_payment',
    'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
    'credit_type', 'co-applicant_credit_type', 'submission_of_application',
    'Region', 'Security_Type'
]

loan_default_data = loan_default_data.drop(columns=['age'])
loan_default_data = encode_categoricals(loan_default_data, categorical_columns)

X = loan_default_data.drop(columns=['Status', 'ID'])
y = loan_default_data['Status']

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training set default rate: {y_train.mean():.4f}")
print(f"Test set default rate: {y_test.mean():.4f}")

model = train_logistic_regression(X_train_scaled, y_train)

print("Model trained successfully")
print(f"Number of coefficients: {len(model.coef_[0])}")

con_matrix, class_report, auc = evaluate_model(model, X_test_scaled, y_test)

print("\nConfusion Matrix:")
print(con_matrix)
print("\nClassification Report:")
print(class_report)
print(f"\nAUC-ROC Score: {auc:.4f}")