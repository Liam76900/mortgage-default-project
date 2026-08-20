import pandas as pd

def check_risk_deciles(model, X_test, y_test):
    predicted_probs = model.predict_proba(X_test)[:, 1]
    results = pd.DataFrame({'predicted_prob': predicted_probs, 'actual': y_test.values})
    results['decile'] = pd.qcut(results['predicted_prob'], 10, labels=False, duplicates='drop')
    return results.groupby('decile')['actual'].mean()