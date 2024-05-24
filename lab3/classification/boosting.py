from functions import readData, printAccuracy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = readData()

param_grid = {
	'n_estimators': [50, 100, 200],
	'learning_rate': [0.01, 0.1, 0.2],
	'max_depth': [3, 5, 7],
}

gb_model = GradientBoostingClassifier()

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)
print(best_model)

y_pred = best_model.predict(X_test)

printAccuracy(y_test, y_pred)
