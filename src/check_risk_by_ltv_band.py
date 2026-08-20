import pandas as pd

def check_risk_by_ltv_band(model, X, scaler):
    X_scaled = scaler.transform(X)
    predicted_probs = model.predict_proba(X_scaled)[:, 1]

    results = pd.DataFrame({
        'LTV': X['LTV'].values,
        'predicted_probability': predicted_probs
    })

    results['LTV_band'] = pd.cut(results['LTV'], bins=[0, 60, 80, 100, 120, 200])
    return results.groupby('LTV_band', observed=True)['predicted_probability'].mean()