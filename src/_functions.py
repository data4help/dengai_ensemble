
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

# %% Plotting predictions results

def plotting_predictions(y_pred_list, y_test, threshold_list, city):

    fig, axs = plt.subplots(figsize=(10, 10))
    for y_pred, threshold_level in zip(y_pred_list, threshold_list):
        axs.plot(y_pred.values, label=f"Threshold level at: {threshold_level}")
    axs.plot(y_test.values, label="True Values", marker="o", linestyle="None")
    axs.legend()
    path = f"{FIGURES_PATH}/{city}_all_threshold_levels.png"
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
