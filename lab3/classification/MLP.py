from functions import printAccuracy, readData
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = readData()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=200, solver='adam', early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
mlp.fit(X_train, y_train)

param_grid = {
    'hidden_layer_sizes': [(3),   (3,3)],
    'activation': ['tanh', 'relu'],
    'alpha': [0.001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_mlp = MLPClassifier(**best_params, max_iter=200, random_state=42)
best_mlp.fit(X_train, y_train)
y_pred = best_mlp.predict(X_test)

printAccuracy(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Raport klasyfikacji:\n", report)

model_params = best_mlp.get_params()
coefs = best_mlp.coefs_
intercepts = best_mlp.intercepts_

print("Parametry modelu:\n", model_params)
print("Współczynniki wag:\n", coefs)
print("Przecięcia:\n", intercepts)
