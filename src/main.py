
"""This file uses classification, regression and smoothing for classifying the number of cases for dengue fever"""

# %% Packages

from tqdm import tqdm
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imblearn_make_pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn import svm

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split

from src._classes import (Encoding, ColumnCorrection, Transformer, Stationarity, Imputer, FeatureCreation,
                          FeatureSelection, TBATSWrapper, ModelSwitcher)
from src._functions import (city_query, winsorize_data, combining_models, plotting_predictions, save_prediction_results)
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

def combination_model(X_train, X_test, y_train, threshold):

    # Classification
    binary_target = y_train >= np.percentile(y_train, threshold)
    clf_gscv.fit(X_train, binary_target)
    clf_pred_train = clf_gscv.best_estimator_.predict(X_train)
    clf_pred_test = clf_gscv.best_estimator_.predict(X_test)

    # Regression
    subset_x_train, subset_y_train = X_train.loc[clf_pred_train, :], y_train[clf_pred_train]
    subset_x_test = X_test.loc[clf_pred_test, :]
    reg_gscv.fit(subset_x_train, subset_y_train)
    if sum(clf_pred_test) != 0:
        reg_y_pred_test = reg_gscv.best_estimator_.predict(subset_x_test).round().astype(int)
    else:
        reg_y_pred_test = []

    # Smoother
    winsorized_y_train = winsorize_data(data=y_train, upper_threshold=threshold)
    smt = TBATSWrapper().fit(X=X_train, y=winsorized_y_train)
    smt_pred_test = smt.predict(X=X_test)

    # Combination models
    total_pred_test = combining_models(clf=clf_pred_test, reg=reg_y_pred_test, smt=smt_pred_test, index_df=X_test)
    return total_pred_test

# %% Testing

test_pred_results = {}
threshold_list = [75, 85, 95]
for city in tqdm(["iq", "sj"]):
    mae_list, y_pred_list = [], []
    for threshold in tqdm(threshold_list):

        # Load data
        X_train_total, X_test_total, y_train_total = city_query(city)
        X_train, X_test, y_train, y_test = train_test_split(X_train_total, y_train_total, test_size=0.2, shuffle=False)

        # Predictions
        y_pred = combination_model(X_train, X_test, y_train, threshold)
        mae = mean_absolute_error(y_test, y_pred)

        mae_list.append(mae)
        y_pred_list.append(y_pred)

    plotting_predictions(y_pred_list, y_test, threshold_list, mae_list, city)
    threshold_level = threshold_list[np.argmin(mae_list)]
    test_pred_results[city] = combination_model(X_train_total, X_test_total, y_train_total, threshold_level)
save_prediction_results(test_pred_results)
