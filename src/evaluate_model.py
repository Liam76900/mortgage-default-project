from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    con_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return con_matrix, class_report, auc