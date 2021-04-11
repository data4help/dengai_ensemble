
"""This file uses classification, regression and smoothing for classifying the number of cases for dengue fever"""

# %% Packages

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imblearn_make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from src._classes import (Encoding, ColumnCorrection, Transformer, Stationarity, Imputer, FeatureCreation,
                          FeatureSelection, TBATSWrapper, ModelSwitcher)
from src._functions import (city_query, find_top_n_obs, winsorize_data, plot_confusion_matrix,
                            combining_models, plot_prediction_results, plot_total, save_prediction_results)
import src._config

# %% General Pipeline settings

# Pipeline settings
cv = 2
N_NEIGHBOURS = 10
DEGREE = 2
MAX_LAG = 4
SIGNIFICANCE_LEVEL = 5 / 100
RS = 42
e_list = [0.01, 0.05, 0.1, 0.25, 0.5]
clf_scoring = "precision"
reg_scoring = "neg_mean_absolute_error"

# %% Pipeline settings

# Classification Pipeline
clf_pipeline = imblearn_make_pipeline(
    Encoding(),
    ColumnCorrection(),
    Stationarity(SIGNIFICANCE_LEVEL=SIGNIFICANCE_LEVEL),
    Transformer(),
    Imputer(n_neighbors=N_NEIGHBOURS),
    FeatureCreation(degree=DEGREE, max_lag=MAX_LAG),
    FeatureSelection(e_list=e_list, scoring=clf_scoring, clf=True),
    SMOTE(random_state=RS),
    ModelSwitcher()
)

clf_parameters = [
    {"modelswitcher__estimator": [svm.SVC(random_state=RS)],
     "featureselection__estimator": [svm.SVC(random_state=RS)]},

    {"modelswitcher__estimator": [LogisticRegression(random_state=RS)],
     "featureselection__estimator": [LogisticRegression(random_state=RS)]},

    {"modelswitcher__estimator": [RandomForestClassifier(random_state=RS)],
     "featureselection__estimator": [RandomForestClassifier(random_state=RS)]},

    {"modelswitcher__estimator": [GradientBoostingClassifier(random_state=RS)],
     "featureselection__estimator": [GradientBoostingClassifier(random_state=RS)]},
]

clf_gscv = GridSearchCV(estimator=clf_pipeline, param_grid=clf_parameters, scoring=clf_scoring, cv=cv)

# Regression Pipeline
reg_pipeline = imblearn_make_pipeline(
    Encoding(),
    ColumnCorrection(),
    Stationarity(SIGNIFICANCE_LEVEL=SIGNIFICANCE_LEVEL),
    Transformer(),
    Imputer(n_neighbors=N_NEIGHBOURS),
    FeatureCreation(degree=DEGREE, max_lag=MAX_LAG, lagged_features=False),
    FeatureSelection(e_list=e_list, scoring=reg_scoring, clf=False),
    ModelSwitcher()
)

reg_parameters = [
    {"modelswitcher__estimator": [LinearRegression()],
     "featureselection__estimator": [LinearRegression()]},

    {"modelswitcher__estimator": [RandomForestRegressor(random_state=RS)],
     "featureselection__estimator": [RandomForestRegressor(random_state=RS)]},

    {"modelswitcher__estimator": [GradientBoostingRegressor(random_state=RS)],
     "featureselection__estimator": [GradientBoostingRegressor(random_state=RS)]}
]

reg_gscv = GridSearchCV(estimator=reg_pipeline, param_grid=reg_parameters, scoring=reg_scoring, cv=cv)

# %% Prediction function

class CombinationModel(BaseEstimator, RegressorMixin):

    def __init__(self, city, reg_pipeline, clf_pipeline, threshold=None):
        self.city = city
        self.reg_pipeline = reg_pipeline
        self.clf_pipeline = clf_pipeline
        self.threshold = threshold

    def fit(self, X_train, y):

        binary_target = find_top_n_obs(target=y, threshold=self.threshold, city=self.city)
        self.clf_pipeline.fit(X_train, binary_target)

        subset_x_train, subset_y_train = X_train.loc[binary_target.values, :], y[binary_target.values]
        self.reg_pipeline.fit(subset_x_train, subset_y_train)

        winsorized_y_train = winsorize_data(data=y, upper_threshold=self.threshold)
        self.smt = TBATSWrapper().fit(X=X_train, y=winsorized_y_train)

    def predict(self, X_test):

        clf_test = self.clf_pipeline.best_estimator_.predict(X_test)
        subset_X_test = X_test.loc[clf_test, :]
        if sum(clf_test) != 0:
            reg_test = self.reg_pipeline.best_estimator_.predict(subset_X_test).round().astype(int)
        else:
            reg_test = []
        smt_test = self.smt.predict(X=X_test)

        total_forecast = combining_models(clf=clf_test, reg=reg_test, smt=smt_test, index_df=X_test)
        return total_forecast

# %% Obtaining results

test_pred_results = {}
param_grid = {"threshold": [95, 97.5]}
tscv = TimeSeriesSplit(n_splits=2)
for city in tqdm(["sj", "iq"]):

    X_train, X_test, y_train = city_query(city)
    estimator = CombinationModel(city=city, reg_pipeline=reg_gscv, clf_pipeline=clf_gscv)
    combination_gscv = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=tscv, scoring=reg_scoring)
    combination_gscv.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(combination_gscv.cv_results_)
    print(cv_results_df.head())

    test_pred_results[city] = {}
    test_pred_results[city]["train"] = combination_gscv.best_estimator_.predict(X_train)
    test_pred_results[city]["test"] = combination_gscv.best_estimator_.predict(X_test)

# %% Plotting test predictions

for city in test_pred_results.keys():
    y_pred = test_pred_results[city].copy()
    plot_total(y_pred, city)
save_prediction_results(test_pred_results)

