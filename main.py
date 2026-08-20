import sys
sys.path.append('src')

import pandas as pd
from data_loader import load_data
from age_data_cleaner import encode_age
from missing_values_cleaner import clean_missing_values
from income_to_loan_ratio import add_income_to_loan_ratio
from categoricals_encoder import encode_categoricals
from data_splitter import split_data
from train_logistic_regression import train_logistic_regression
from evaluate_model import evaluate_model
from feature_scaler import scale_features
from feature_importance_calculator import get_feature_importance
from calibration_check import check_calibration
from check_risk_deciles import check_risk_deciles
from check_risk_by_ltv_band import check_risk_by_ltv_band
from decile_chart_plot import plot_decile_chart
from ltv_band_chart_plot import plot_ltv_band_chart
from roc_curve_plot import plot_roc_curve
from feature_importance_plot import plot_feature_importance

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

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training set default rate: {y_train.mean():.4f}")
print(f"Test set default rate: {y_test.mean():.4f}")

feature_names = X_train.columns.tolist()

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

model_baseline = train_logistic_regression(X_train_scaled, y_train, class_weight=None)
model_balanced = train_logistic_regression(X_train_scaled, y_train, class_weight="balanced")

con_matrix_baseline, class_report_baseline, auc_baseline = evaluate_model(model_baseline, X_test_scaled, y_test)
con_matrix_balanced, class_report_balanced, auc_balanced = evaluate_model(model_balanced, X_test_scaled, y_test)


print("\nBaseline Model")
print(con_matrix_baseline)
print(class_report_baseline)
print(f"AUC: {auc_baseline:.4f}")

print("\nBalanced Model")
print(con_matrix_balanced)
print(class_report_balanced)
print(f"AUC: {auc_balanced:.4f}")

importance_df = get_feature_importance(model_balanced, feature_names)

print("\nTop 10 features increasing default risk:")
print(importance_df.head(10))

print("\nTop 10 features decreasing default risk:")
print(importance_df.tail(10))

avg_pred_baseline, actual_baseline = check_calibration(model_baseline, X_test_scaled, y_test)
avg_pred_balanced, actual_balanced = check_calibration(model_balanced, X_test_scaled, y_test)

print(f"\nBaseline — Average predicted probability: {avg_pred_baseline:.4f}, Actual rate: {actual_baseline:.4f}")
print(f"Balanced — Average predicted probability: {avg_pred_balanced:.4f}, Actual rate: {actual_balanced:.4f}")

decile_check_baseline = check_risk_deciles(model_baseline, X_test_scaled, y_test)
decile_check_balanced = check_risk_deciles(model_balanced, X_test_scaled, y_test)

print("Baseline model — actual default rate by predicted risk decile:")
print(decile_check_baseline)

print("\nBalanced model — actual default rate by predicted risk decile:")
print(decile_check_balanced)

ltv_check_balanced = check_risk_by_ltv_band(model_balanced, X_test, scaler)
print("\nBalanced model — average predicted probability by LTV band:")
print(ltv_check_balanced)

plot_decile_chart(decile_check_baseline, decile_check_balanced)
plot_ltv_band_chart(ltv_check_balanced)
plot_roc_curve(model_baseline, X_test_scaled, y_test, model_name='Baseline')
plot_roc_curve(model_balanced, X_test_scaled, y_test, model_name='Balanced')
plot_feature_importance(importance_df)