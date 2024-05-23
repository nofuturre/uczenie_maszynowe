from functions import readData, printAccuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV
)

X_train, X_test, y_train, y_test = readData()

model = GaussianNB()

cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=3, 
                                    random_state=999)

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gs_NB = GridSearchCV(estimator=model, 
                     param_grid=params_NB, 
                     cv=cv_method,
                     verbose=1, 
                     scoring='accuracy')

Data_transformed = PowerTransformer().fit_transform(X_test)

gs_NB.fit(Data_transformed, y_test)

print(gs_NB.best_params_)
print(gs_NB.best_score_)

# predict the target on the test dataset
y_pred = gs_NB.predict(Data_transformed)

printAccuracy(y_test, y_pred)
