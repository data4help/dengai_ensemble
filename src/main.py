
"""This file uses classification, regression and smoothing for classifying the number of cases for dengue fever"""

# %% Packages

import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imblearn_make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from src._classes import (Encoding, ColumnCorrection, Transformer, Stationarity, Imputer, FeatureCreation,
                          FeatureSelection, TBATSWrapper, ModelSwitcher)
from src._functions import (city_query, find_top_n_obs, winsorize_data, plot_confusion_matrix,
                            combining_models, plot_prediction_results, plot_total)
import src._config

# %% General Pipeline settings

# Pipeline settings
cv = 3
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
     "modelswitcher__estimator__C": [0.1, 1, 5],
     "featureselection__estimator": [svm.SVC(random_state=RS)],
     "smote__sampling_strategy": [0.5, 1]},

    {"modelswitcher__estimator": [LogisticRegression(random_state=RS)],
     "modelswitcher__estimator__C": [0.1, 1, 2],
     "featureselection__estimator": [LogisticRegression(random_state=RS)],
     "smote__sampling_strategy": [0.5, 1]},

    {"modelswitcher__estimator": [RandomForestClassifier(random_state=RS)],
     "modelswitcher__estimator__min_samples_leaf": [1, 5, 10],
     "featureselection__estimator": [RandomForestClassifier(random_state=RS)],
     "smote__sampling_strategy": [0.5, 1]},

    {"modelswitcher__estimator": [GradientBoostingClassifier(random_state=RS)],
     "modelswitcher__estimator__learning_rate": [0.05, 0.1, 0.2],
     "featureselection__estimator": [GradientBoostingClassifier(random_state=RS)],
     "smote__sampling_strategy": [0.5, 1]},
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
     "modelswitcher__estimator__min_samples_leaf": [1, 5, 10, 25, 50],
     "featureselection__estimator": [RandomForestRegressor(random_state=RS)]},

    {"modelswitcher__estimator": [GradientBoostingRegressor(random_state=RS)],
     "modelswitcher__estimator__min_samples_leaf": [1, 5, 10],
     "modelswitcher__estimator__max_depth": [3, 5, 10],
     "modelswitcher__estimator__learning_rate": [0.1, 0.2],
     "featureselection__estimator": [GradientBoostingRegressor(random_state=RS)]}
]

reg_gscv = GridSearchCV(estimator=reg_pipeline, param_grid=reg_parameters, scoring=reg_scoring, cv=cv)

# %% Prediction function

def make_predictions(city, threshold):

    # Extract the data
    X_train, X_test, y_train = city_query(city)

    # Classification
    binary_target = find_top_n_obs(target=y_train, threshold=threshold, city=city)
    clf_gscv.fit(X_train, binary_target)
    clf_pred_train = clf_gscv.best_estimator_.predict(X_train)
    clf_pred_test = clf_gscv.best_estimator_.predict(X_test)
    plot_confusion_matrix(y_true=binary_target, y_pred=clf_pred_train, city=city, threshold=threshold)

    # Regression
    subset_x_train, subset_y_train = X_train.loc[clf_pred_train, :], y_train[clf_pred_train]
    subset_x_test = X_test.loc[clf_pred_test, :]
    reg_gscv.fit(subset_x_train, subset_y_train)
    reg_y_pred_train = reg_gscv.best_estimator_.predict(subset_x_train).round().astype(int)
    if sum(clf_pred_test) != 0:
        reg_y_pred_test = reg_gscv.best_estimator_.predict(subset_x_test).round().astype(int)
    else:
        reg_y_pred_test = []

    # Smoother
    winsorized_y_train = winsorize_data(data=y_train, upper_threshold=threshold)
    smt = TBATSWrapper().fit(X=X_train, y=winsorized_y_train)
    smt_pred_train = smt.in_sample_predict()
    smt_pred_test = smt.predict(X=X_test)

    # Combination models
    total_pred_train = combining_models(clf=clf_pred_train, reg=reg_y_pred_train, smt=smt_pred_train, index_df=X_train)
    total_pred_test = combining_models(clf=clf_pred_test, reg=reg_y_pred_test, smt=smt_pred_test, index_df=X_test)
    mae = mean_absolute_error(y_true=y_train, y_pred=total_pred_train)

    return total_pred_train, total_pred_test, mae, clf_gscv.best_params_, reg_gscv.best_params_

# %% Obtaining results

test_pred_results = {}
threshold_list = [90, 95, 99]
for city in tqdm(["iq", "sj"]):
    train_data, test_data, mae_list, clf_list, reg_list = [], [], [], [], []
    for threshold in tqdm(threshold_list):
        y_pred_train, y_pred_test, mae, clf_dict, reg_dict = make_predictions(city, threshold)
        train_data.append(y_pred_train)
        test_data.append(y_pred_test)
        mae_list.append(mae)
        clf_list.append(clf_dict)
        reg_list.append(reg_dict)

    best_mae_argmin = plot_prediction_results(train_data=train_data, threshold_list=threshold_list,
                                              mae_list=mae_list, city=city)
    test_pred_results[city] = test_data[best_mae_argmin]
    test_data[best_mae_argmin].to_csv(f"/Users/PM/Documents/projects/dengai_ensemble/data/predictions/{city}.csv")
    print(clf_list[best_mae_argmin])
    print(reg_list[best_mae_argmin])

# %% Plotting test predictions

for city in test_pred_results.keys():
    y_pred = test_pred_results[city].copy()
    plot_total(y_pred, city)

# %% Saving predictions

_, _, y_

test_pred_results
