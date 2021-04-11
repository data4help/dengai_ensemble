
# %% Packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix

RAW_PATH = f"{os.getcwd()}/data"
FIGURES_PATH = f"{os.getcwd()}/reports/figures"

# %% Load the data

def load_data():
    """Loading the data"""
    features_train = pd.read_csv(f"{RAW_PATH}/dengue_features_train.csv")
    features_test = pd.read_csv(f"{RAW_PATH}/dengue_features_test.csv")
    target_train_data = pd.read_csv(f"{RAW_PATH}/dengue_labels_train.csv")
    return features_train, features_test, target_train_data

# %% Helper function

def city_query(city_name):
    """This function extracts a specific city out of the dataframe which contains both"""
    features_train, features_test, target_train = load_data()
    features_train_city = features_train.query("city==@city_name")
    features_test_city = features_test.query("city==@city_name")

    target_train_city = target_train.query("city==@city_name")
    target_variable_city = target_train_city.loc[:, "total_cases"]

    features_train_city.index = features_train_city.loc[:, "week_start_date"]
    features_test_city.index = features_test_city.loc[:, "week_start_date"]

    return features_train_city, features_test_city, target_variable_city

# %% Creating the binary target

def find_top_n_obs(target, threshold, city):
    """This function takes in the target variable and returns a boolean of whether the observation belongs to the
    top X percent."""
    top_cases_bool = target >= np.percentile(target, threshold)
    top_cases_numbers = target[top_cases_bool]
    fig, axs = plt.subplots(figsize=(10, 10))
    target.plot.line(ax=axs)
    top_cases_numbers.plot(ax=axs, style="o")
    path = f"{FIGURES_PATH}/top_{threshold}_levels_{city}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    return top_cases_bool

# %% Winsorizer

def winsorize_data(data, upper_threshold):
    """This function cuts the data at the upper threshold"""
    one_minus_upper_bound = 1-(upper_threshold/100)
    cutted_data = winsorize(data, limits=[0, one_minus_upper_bound], inclusive=(True, False)).data
    return cutted_data

# %% Combination models

def combining_models(clf, reg, smt, index_df):
    """This function is combining all different methods and ensures that negatives values are set to zero"""
    total_pred = smt.copy()
    total_pred[clf] = reg
    total_pred[total_pred < 0] = 0
    total_pred_df = pd.DataFrame(data=total_pred, columns=["predictions"], index=index_df.index)
    return total_pred_df

# %% Plotting prediction results

def plot_prediction_results(train_data, threshold_list, mae_list, city):

    """This function takes all the predictions which come from the different thresholds."""
    _, _, y_train = city_query(city)
    n_train = len(train_data)
    fig, axs = plt.subplots(figsize=(15*n_train, 10), ncols=n_train)
    axs = axs.ravel()
    for i, (data, threshold, mae) in enumerate(zip(train_data, threshold_list, mae_list)):
        axs[i].plot(data.values, label=f"Threshold at level {threshold} - MAE: {round(mae, 2)}")
        axs[i].plot(y_train.values, label="True Values", linestyle="None", marker="o", alpha=0.2)
        axs[i].legend(loc="best")
    path = f"{FIGURES_PATH}/different_thresholds_{city}.png"
    fig.savefig(fname=path, bbox_inches="tight")

    smallest_mae = np.argmin(mae_list)
    return smallest_mae

# %% Plotting confusion matrix

def plot_confusion_matrix(y_true, y_pred, city, threshold):
    """This function plots a confusion matrix for the different classifiers"""
    confusion_matrix_array = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.heatmap(data=confusion_matrix_array, annot=True, ax=axs, fmt="d")
    axs.set_ylabel("True Values")
    axs.set_xlabel("Predicted Values")
    path = f"{FIGURES_PATH}/confusion_matrix/{city}_{threshold}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()

# %% Plot entire time series with predictions

def plot_total(y_pred, city):
    _, _, y_train = city_query(city)
    y_train.reset_index(drop=True, inplace=True)

    new_index = list(range(len(y_train), len(y_train) + len(y_pred)))
    y_pred_train = y_pred["train"]
    y_pred_train.index = y_train.index

    fig, axs = plt.subplots(figsize=(20, 10))
    axs.plot(y_train, color="blue", label="True values")
    axs.plot(y_pred_train, color="red", label="Predictions")
    axs.legend()
    path = f"{FIGURES_PATH}/{city}_total_predictions.png"
    fig.savefig(path, bbox_inches="tight")

# %% Save the prediction results

def save_prediction_results(pred_dict):
    _, test_data, _ = load_data()
    prediction_df = test_data.loc[:, ["city", "year", "weekofyear"]]
    prediction_df.loc[:, "total_cases"] = np.nan

    for (city, predictions) in pred_dict.items():
        bool_city = prediction_df.loc[:, "city"] == city
        prediction_df.loc[bool_city, "total_cases"] = predictions.values

    todays_date = datetime.today().strftime("%Y%m%d")
    prediction_df.loc[:, "total_cases"] = prediction_df.loc[:,"total_cases"].astype(int)
    prediction_df.to_csv(f"{RAW_PATH}/predictions/combination_{todays_date}.csv", index=False)
