
"""This file aims to classify first the top five percent of cases (the spikes) and then uses this variable as
a dummy variable when predicting the amount of cases"""

# %% Packages, Pathing and loading the data

import warnings
import numpy as np
import pandas as pd
from tbats import TBATS
from scipy import signal as sig
from statsmodels.tsa.stattools import adfuller
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# %% Pipeline classes

class Encoding(BaseEstimator, TransformerMixin):
    """This class is creating dummy variables out of the month variable. Further, it is dropping the time variable
    week start date and the year. The year is dropped given that this variable is not useful and will
    not be stationary. Furthermore, we mean encode the weekofyear variable."""

    def __init__(self):
        pass

    def fit(self, X, y):
        # Get mean number of cases for weekofyear
        weekofyear_df = X.loc[:, ["weekofyear"]].copy()
        weekofyear_df.loc[:, "target"] = y.values
        self.weekofyear_dict = weekofyear_df.groupby("weekofyear")["target"].mean().to_dict()

        # Making sure that we have all weeks of the year within the data
        week_within_a_year = 53
        keys_present = self.weekofyear_dict.keys()
        overall_average = np.mean(list(self.weekofyear_dict.values()))
        for i in np.arange(1, week_within_a_year+1):
            if i not in keys_present:
                self.weekofyear_dict[i]: overall_average

        return self

    def transform(self, X):

        # Creating monthly dummy variables from week start date
        week_start_date = pd.to_datetime(X.loc[:, "week_start_date"])
        X.loc[:, "month"] = week_start_date.dt.month.astype(object)
        subset_X = X.drop(columns=["week_start_date", "year"])
        dummy_X = pd.get_dummies(subset_X, drop_first=True)

        # Applying the mean encoding for the weekofyear variable
        dummy_X.loc[:, "weekofyear"] = X.loc[:, "weekofyear"].map(self.weekofyear_dict)

        return dummy_X

class ColumnCorrection(BaseEstimator, TransformerMixin):
    """This class is necessary to ensure that all columns we created in train are also showing up in the test data."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.train_columns = X.columns
        return self

    def transform(self, X):
        in_train_but_not_in_test = list(set(self.train_columns) - set(X.columns))
        if len(in_train_but_not_in_test) != 0:
            for col in in_train_but_not_in_test:
                X.loc[:, col] = 0
        X = X.loc[:, self.train_columns]
        return X

class Stationarity(BaseEstimator, TransformerMixin):
    """This class is checking for stationarity using the augmented dickey fuller test."""

    def __init__(self, SIGNIFICANCE_LEVEL):
        self.SIGNIFICANCE_LEVEL = SIGNIFICANCE_LEVEL

    def check_stationarity(self, series):
        """This method conducts two stationary tests, namely the ADF and KPSS test"""
        no_nan_series = series.dropna()
        adf_stationary = adfuller(x=no_nan_series, regression="c", autolag="AIC")[1] < self.SIGNIFICANCE_LEVEL
        return adf_stationary

    def adjust_series(self, series, adf_stationary):
        """This method takes care of any adjustments needed to be done for the different cases of nonstationarity"""
        series.dropna(inplace=True)
        series = series.diff()
        adf_stationary = self.check_stationarity(series=series)
        return series, adf_stationary

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """This method takes all columns which are not time dependent and loops over them to check for stationarity.
        This can only be done if there is a minimum amount of observations which has to be calculated before."""

        try:
            non_time_columns = [x for x in X.columns if not x.startswith("month_") and "weekofyear" not in x]
            for col in non_time_columns:
                series = X.loc[:, col].copy()
                adf_stationary = self.check_stationarity(series)
                while not adf_stationary:
                    series, adf_stationary = self.adjust_series(series=series, adf_stationary=adf_stationary)
                    X.loc[:, col] = series.copy()
        except ValueError:
            pass
        return X

class Transformer(BaseEstimator, TransformerMixin):
    """This class is MinMax scaling all variables using trainings data and applying that on the test data."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.columns = X.columns
        scaler = MinMaxScaler()
        self.fitted_scaler = scaler.fit(X)
        return self

    def transform(self, X):
        scaled_X_array = self.fitted_scaler.transform(X)
        scaled_X_df = pd.DataFrame(data=scaled_X_array, columns=self.columns)
        return scaled_X_df

class Imputer(BaseEstimator, TransformerMixin):
    """This class is using a KNN imputation method. In this simpler version we are choosing
    a fixed number of neighbors"""

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.columns = X.columns
        impute = KNNImputer(n_neighbors=self.n_neighbors)
        self.fitted_imputer = impute.fit(X)
        return self

    def transform(self, X):
        imputed_X_array = self.fitted_imputer.transform(X)
        imputed_X_df = pd.DataFrame(data=imputed_X_array, columns=self.columns)
        return imputed_X_df

class FeatureCreation(BaseEstimator, TransformerMixin):
    """Here we create lagged versions of the variables, polynomials and sums."""

    def __init__(self, degree, max_lag, lagged_features=True, polynomial_features=True):
        self.degree = degree
        self.max_lag = max_lag
        self.lagged_features = lagged_features
        self.polynomial_features = polynomial_features

    def fit(self, X, y=None):
        # Saving the last n rows of training in order to append those for the test data
        self.last_train_rows = X.tail(self.max_lag)
        self.train_last_date = X.index[-1]
        return self

    def transform(self, X):

        # Appending last training rows to test data
        if X.index[0] > self.train_last_date:
            X.append(self.last_train_rows)

        non_time_columns = [x for x in X.columns if not x.startswith("month_") and "weekofyear" not in x]
        X_wo_dummy = X.loc[:, non_time_columns]

        # Lagged terms
        if self.lagged_features:
            lagged_data = pd.concat([X_wo_dummy.shift(i) for i in range(1, self.max_lag + 1)], axis=1)
            lagged_columns = [f"{column}_lag{i}" for i in range(1, self.max_lag + 1) for column in non_time_columns]
            lagged_data.columns = lagged_columns
            X = pd.concat([X, lagged_data], axis=1)

        # Interaction terms
        if self.polynomial_features:
            poly_data = X.loc[:, non_time_columns].copy()
            poly = PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=False)
            poly_data_array = poly.fit_transform(poly_data)
            poly_feature_names = poly.get_feature_names(non_time_columns)
            poly_data_df = pd.DataFrame(data=poly_data_array, columns=poly_feature_names)
            poly_wo_initial_features = poly_data_df.loc[:, [x for x in poly_feature_names if x not in non_time_columns]]
            X = pd.concat([X, poly_wo_initial_features], axis=1)

        X.fillna(method="bfill", inplace=True)
        return X

class FeatureSelection(BaseEstimator, TransformerMixin):
    """Using three selection mechanisms for regression or classification to figure out how important which
    features is"""

    def __init__(self, e_list, scoring, clf, estimator=None):
        self.e_list = e_list
        self.scoring = scoring
        self.clf = clf
        self.estimator = estimator

    def _importance_df_creator(self, importances):
        """This method takes the importances of each method and ranks them by their importance"""
        importances_df = pd.DataFrame({"feature_names": self.X_columns, "importances": importances})
        importances_df.loc[:, "rank"] = importances_df.loc[:, "importances"].rank(method="dense", ascending=False)
        importances_df.sort_values(by="rank", inplace=True)
        return importances_df

    def _svm_feature_selection(self, model, X, y):
        model.fit(X, y)
        svm_coef = abs(model.coef_[0])
        importances_df = self._importance_df_creator(svm_coef)
        importances_df.name = "SupportVectorMachine"
        return importances_df

    def _linear_feature_selection(self, model, X, y):
        model.fit(X, y)
        lgr_coef = abs(model.coef_[0])
        importances_df = self._importance_df_creator(lgr_coef)
        importances_df.name = "LinearModel"
        return importances_df

    def _rfr_feature_selection(self, model, X, y):
        model.fit(X, y)
        rfr_fi = model.feature_importances_
        importances_df = self._importance_df_creator(rfr_fi)
        importances_df.name = "RandomForest"
        return importances_df

    def _clustering_features(self, *df_columns, X):
        """This method takes the subset of features and uses an Affinity Propagation to assign each feature to a
        cluster. In the end only the highest ranked feature of each cluster is considered and extracted."""
        relevant_feature_list = []
        cluster_algo = AffinityPropagation(random_state=42, max_iter=2_500)
        for df in df_columns:
            subset_df = df.iloc[:self.n_keep, :].copy()
            chosen_features = subset_df.loc[:, "feature_names"].tolist()
            subset_feature_data = X.loc[:, chosen_features]
            feature_cluster = cluster_algo.fit_predict(subset_feature_data.T)
            subset_df.loc[:, "cluster"] = feature_cluster

            for cluster in set(feature_cluster):
                cluster_df = subset_df.query("cluster==@cluster")
                lowest_rank = cluster_df.loc[:, "rank"].min()
                relevant_features = cluster_df.query("rank==@lowest_rank").loc[:, "feature_names"].tolist()
                relevant_feature_list.extend(relevant_features)
        no_duplicates_relevant_features_list = list(set(relevant_feature_list))
        return no_duplicates_relevant_features_list

    def _adjusted_score(self, cv_scores, nfeatures, nobs):
        """This function adjusts the resulting cv scores by adjusting it by the number of
        observations and features used"""
        scores = np.mean(cv_scores)
        adj_score = 1 - ((1-scores) * (nobs-1) / (nobs-nfeatures-1))
        return adj_score

    def _choose_best_features(self, X, y, e_list):
        """This method initiates the calculation of the importances by each feature. Afterwards we are looping over
        how many features are dropped and calculate the performance of the prediction model. This is done by using the
        same estimator the final model is using as well."""
        cv = 5
        nobs = len(X)
        list_scoring_results, list_selected_features = [], []

        if self.clf:
            svm_model = svm.SVC(kernel="linear", random_state=42)
            linear_model = LogisticRegression(max_iter=5_000, random_state=42)
            rfr_model = RandomForestClassifier(random_state=42)
        else:
            svm_model = svm.SVR(kernel="linear")
            linear_model = LinearRegression()
            rfr_model = RandomForestRegressor(random_state=42)
            self.scoring = "r2"

        svm_columns_df = self._svm_feature_selection(svm_model, X, y)
        lgr_columns_df = self._linear_feature_selection(linear_model, X, y)
        rfr_columns_df = self._rfr_feature_selection(rfr_model, X, y)

        for e in e_list:
            self.n_keep = int(len(self.X_columns) * e)
            selected_features = self._clustering_features(svm_columns_df, lgr_columns_df, rfr_columns_df, X=X)

            list_selected_features.append(selected_features)
            subset_X = X.loc[:, selected_features]
            cv_scores = cross_val_score(self.estimator, subset_X, y, cv=cv, scoring=self.scoring)

            nfeatures = len(subset_X.columns)
            adj_score = self._adjusted_score(cv_scores, nfeatures, nobs)
            list_scoring_results.append(adj_score)

        highest_adj_score_position = np.argmax(list_scoring_results)
        best_features = list_selected_features[highest_adj_score_position]
        return best_features

    def fit(self, X, y=None):
        self.X_columns = X.columns
        self.best_features = self._choose_best_features(X, y, self.e_list)
        return self

    def transform(self, X):
        subset_X = X.loc[:, self.best_features]
        return subset_X

# %% Models wrapper

class ModelSwitcher(BaseEstimator, RegressorMixin):
    """This class allows to use any kind of sklearn prediction model and simply insert the models we would like
    to try out as a hpyer-parameter."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, X, y=None):
        pred = self.estimator.predict(X)
        return pred

# %% Smoother

class TBATSWrapper(BaseEstimator, RegressorMixin):
    """This class fits a TBATS around our timeseries. This is done by first finding the driving frequencies, using
    a fourier transform. Afterwards a TBATS, which can handle multiple seaosnalities is fitted on these seasonalities
    and target variable."""
    def __init__(self):
        pass

    def _finding_driving_frequencies(self, signal):

        fft = np.fft.fft(signal)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(signal))

        left_magnitude = magnitude[:int(len(magnitude)/2)]
        left_frequency = frequency[:int(len(frequency)/2)]

        prominence = np.median(left_magnitude) + 2 * np.std(left_magnitude)
        peaks = sig.find_peaks(left_magnitude[left_frequency >= 0], prominence=prominence)[0]
        peak_freq = left_frequency[peaks]
        list_frequencies = (1/peak_freq).tolist()

        return list_frequencies

    def fit(self, X, y):
        list_frequencies = self._finding_driving_frequencies(signal=y)
        estimator = TBATS(seasonal_periods=list_frequencies, n_jobs=True)
        self.fitted_model = estimator.fit(y)
        return self

    def predict(self, X):
        y_pred = self.fitted_model.forecast(steps=len(X))
        y_pred = y_pred.round().astype(int)
        y_pred[y_pred < 0] = 0
        return y_pred
