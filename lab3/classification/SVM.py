from functions import printAccuracy, readData
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = readData()

svm = SVC()
svm.fit(X_train, y_train)

param_grid = {
    'C': [1],
    'gamma': [0.01],
    'kernel': ['linear']
}

grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_svm = SVC(**best_params)
best_svm.fit(X_train, y_train)

y_pred = best_svm.predict(X_test)

printAccuracy(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Raport klasyfikacji:\n", report)
model_params = best_svm.get_params()
print("Parametry modelu:\n", model_params)
