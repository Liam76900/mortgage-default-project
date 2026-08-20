def check_calibration(model, X_test, y_test):
    predicted_probabilities = model.predict_proba(X_test)[:, 1]
    average_predicted = predicted_probabilities.mean()
    actual_rate = y_test.mean()
    return average_predicted, actual_rate