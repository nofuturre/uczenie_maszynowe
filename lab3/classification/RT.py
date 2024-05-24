from functions import readData, printAccuracy
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = readData()

X_train = X_train[:20]
X_test = X_test[:5]
y_test = y_test[:5]
y_train = y_train[:20]

rf_params = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_params, cv=5)
rf_grid_search.fit(X_train, y_train)

print(rf_grid_search.best_params_)

y_pred = rf_grid_search.predict(X_test)

printAccuracy(y_test, y_pred)