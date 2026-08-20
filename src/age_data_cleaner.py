def encode_age(loan_default_data):
    age_mapping = {
        '<25': 22,
        '25-34': 29.5,
        '35-44': 39.5,
        '45-54': 49.5,
        '55-64': 59.5,
        '65-74': 69.5,
        '>74': 80
    }
    loan_default_data['age_numeric'] = loan_default_data['age'].map(age_mapping)
    return loan_default_data