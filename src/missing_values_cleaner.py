def clean_missing_values(loan_default_data):

    numeric_fill_median = [
        'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges',
        'dtir1', 'income', 'property_value', 'LTV', 'age_numeric'
    ]
    for col in numeric_fill_median:
        median_value = loan_default_data[col].median()
        loan_default_data[col] = loan_default_data[col].fillna(median_value)

    categorical_fill_mode = [
        'loan_limit', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization',
        'submission_of_application'
    ]
    for col in categorical_fill_mode:
        mode_value = loan_default_data[col].mode()[0]
        loan_default_data[col] = loan_default_data[col].fillna(mode_value)

    loan_default_data = loan_default_data.dropna(subset=['term'])

    return loan_default_data