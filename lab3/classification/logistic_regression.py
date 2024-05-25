import numpy as np
from functions import printAccuracy, readData
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = readData()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

param_grid = {
    'logreg__C': np.logspace(-4, 4, 20),
    'logreg__solver': ['lbfgs', 'liblinear', 'saga']
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Najlepsze hiperparametry: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
printAccuracy(y_test, y_pred)
