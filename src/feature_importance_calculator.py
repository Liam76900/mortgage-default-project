import pandas as pd

def get_feature_importance(model, feature_names):
    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })
    importance_df = importance_df.sort_values(by="coefficient", ascending=False)
    return importance_df