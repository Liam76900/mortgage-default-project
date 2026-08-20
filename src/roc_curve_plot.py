from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(model, X_test, y_test, model_name='Model'):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f'{model_name}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend()
    plt.savefig(f'outputs/roc_curve_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()