from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from lab3.classification.functions import printAccuracy, readData

X_train, X_test, y_train, y_test = readData()

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

param_grid = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'store_covariance': [True, False],
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
}

grid_search = GridSearchCV(LinearDiscriminantAnalysis(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_lda = LinearDiscriminantAnalysis(**best_params)
best_lda.fit(X_train, y_train)
y_pred = best_lda.predict(X_test)

printAccuracy(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Raport klasyfikacji:\n", report)

model_params = best_lda.get_params()
coefficients = best_lda.coef_
intercepts = best_lda.intercept_

print("Parametry modelu:\n", model_params)
print("Współczynniki:\n", coefficients)
print("Przecięcia:\n", intercepts)
