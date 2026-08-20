from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, class_weight=None):
    model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model