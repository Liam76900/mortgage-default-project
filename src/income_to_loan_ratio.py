def add_income_to_loan_ratio(loan_default_data):
    loan_default_data['income_to_loan_ratio'] = (
        loan_default_data['income'] / loan_default_data['loan_amount']
    )
    return loan_default_data