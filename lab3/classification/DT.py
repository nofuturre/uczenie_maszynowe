from functions import readData, printAccuracy
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X_train, X_test, y_train, y_test = readData()

dt = DecisionTreeClassifier(random_state=42)

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train, y_train)

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df.head())
print(score_df.nlargest(5,"mean_test_score"))
print(grid_search.best_estimator_)

dt_best = grid_search.best_estimator_
y_pred = dt_best.predict(X_test)

printAccuracy(y_test, y_pred)

