import warnings

from functions import printAccuracy, readData
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = readData()

bagging = BaggingClassifier(random_state=42)

bagging.fit(X_train, y_train)
param_grid = {
    'n_estimators': [5, 10],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False]
}

grid_search = GridSearchCV(bagging, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_bagging = BaggingClassifier(**best_params, random_state=42)
best_bagging.fit(X_train, y_train)
y_pred = best_bagging.predict(X_test)

printAccuracy(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Raport klasyfikacji:\n", report)
model_params = best_bagging.get_params()
print("Parametry modelu:\n", model_params)
