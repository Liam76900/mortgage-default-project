import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(importance_df, top_n=10):
    top_features = pd.concat([importance_df.head(top_n), importance_df.tail(top_n)])

    plt.figure(figsize=(9,7))
    colors = ['crimson' if c > 0 else 'steelblue' for c in top_features['coefficient']]
    plt.barh(top_features['feature'], top_features['coefficient'], color=colors)
    plt.xlabel('Coefficient (positive = increases risk, negative = decreases risk)')
    plt.title('Top Features Driving Default Risk')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()