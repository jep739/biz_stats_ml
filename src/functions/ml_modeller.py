import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
import getpass
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import geopandas as gpd
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
import shap


import sys

sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
import kommune_translate

fs = FileClient.get_gcs_file_system()
import numpy as np


import warnings

warnings.filterwarnings("ignore")

import math

# good_df = ao.rette_bedrifter(good_df)

import input_data
import create_datafiles

from joblib import Parallel, delayed
import multiprocessing

import time

def hente_training_data():
    
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/temp/training_data.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    training_data = table.to_pandas()
        
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/temp/imputatable_df.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    imputatable_df = table.to_pandas()
    
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g2021/statistikkfil_foretak_pub.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    foretak_pub = table.to_pandas()
    
    foretak_pub['n3'] = foretak_pub['naring_f'].str[:4]
    foretak_pub['n2'] = foretak_pub['naring_f'].str[:2]
    
    foretak_pub = foretak_pub[foretak_pub['n2'].isin(['45', '46', '47'])]
    foretak_pub = foretak_pub[['n3',
                             'bearbeidingsverdi',
                             'produktinnsats',
                             'produksjonsverdi',
                             'omsetning',
                             'sysselsetting_syss',
                             'ts_forbruk',
                            'ts_avanse',
                            'ts_salgsint',
                            'ts_vikarutgifter',
                            'ts_byggvirk',
                            'ts_varehan',
                            'ts_anlegg',
                            'ts_tjeneste',
                            'ts_industri',
                            'ts_agentur',
                            'ts_detalj',
                            'ts_engros',
                            'ts_internet_salg',
                            'ts_annet',
                            'nopost_lonnskostnader',
                            'nopost_driftskostnader',
                            'nopost_driftsresultat',
                            'nopost_driftsinntekter',
                            'saldo_kjop_p0580']]
        
    return training_data, imputatable_df, foretak_pub

def xgboost_model(training_df, scaler, df_estimeres, GridSearch=True):
    """
    Trains an XGBoost model for predicting new_oms values with an optional GridSearch for hyperparameter tuning.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import matplotlib.pyplot as plt
    import shap

    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    # Drop rows with NaN values in the target column
    df = df.dropna(subset=['new_oms'])
    
    # Convert specified columns to category type
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target variable
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define categorical and numerical features
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast", 
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),  # Apply scaling to numerical features
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),  # One-hot encoding for categorical features
        ]
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform the training and testing data
    X_train_transformed = preprocessor.transform(X_train).toarray()
    X_test_transformed = preprocessor.transform(X_test).toarray()

    if GridSearch:
        # Define the model
        regressor = xgb.XGBRegressor(eval_metric="rmse", random_state=42)

        # Define parameter grid for GridSearch
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Perform GridSearch with cross-validation
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train_transformed, y_train)

        # Print best parameters
        print("Best parameters found by GridSearch:", grid_search.best_params_)

        # Use best estimator from grid search
        regressor = grid_search.best_estimator_
    else:
        # Define the model with default parameters
        regressor = xgb.XGBRegressor(eval_metric="rmse", random_state=42)

        # Train the model
        eval_set = [(X_train_transformed, y_train), (X_test_transformed, y_test)]
        regressor.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)

    # Evaluate the model
    y_pred = regressor.predict(X_test_transformed)

    # Check for negative values in predictions
    negative_indices = np.where(y_pred < 0)[0]
    negative_predictions = y_pred[y_pred < 0]

    if len(negative_predictions) > 0:
        print("Number of negative predictions:", len(negative_predictions))
    else:
        print("No negative predictions found.")
        
    # Set negative predictions to zero
    y_pred = np.maximum(y_pred, 0)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)  # Calculate Mean Absolute Error
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)
    print("Mean Absolute Error:", mae)

    # Plot the learning history
    results = regressor.evals_result()
    epochs = len(results["validation_0"]["rmse"])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, results["validation_0"]["rmse"], label="Train")
    plt.plot(x_axis, results["validation_1"]["rmse"], label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("XGBoost Learning History")
    plt.show()

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Tree textual representation
    booster = regressor.get_booster()
    with open("dump.raw.txt", "w") as f:
        f.write("\n".join(booster.get_dump()))
    print(booster.get_dump()[0])  # Print the first tree

    # SHAP values
    explainer = shap.TreeExplainer(regressor, X_train_transformed)
    shap_values = explainer.shap_values(X_test_transformed)

    # Get feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # Summary plot of SHAP values
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

    # Force plot for a single prediction (e.g., the first instance)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], X_test_transformed[0], feature_names=feature_names)

    # Find the correct index for the feature "verdi"
    verdi_index = list(feature_names).index("num__new_oms_trendForecast")

    # Dependence plot to show the effect of a single feature across the dataset
    shap.dependence_plot(verdi_index, shap_values, X_test_transformed, feature_names=feature_names)

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)

    # Ensure no negative predictions
    imputed_df['predicted_oms'] = imputed_df['predicted_oms'].clip(lower=0)
    imputed_df['predicted_oms'] = imputed_df['predicted_oms'].astype(float)
    
    return imputed_df



def knn_model(training_df, scaler, df_estimeres, GridSearch=True):
    """
    Trains a K-Nearest Neighbors model for predicting new_oms values with an optional GridSearch for hyperparameter tuning.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.neighbors import KNeighborsRegressor
    import matplotlib.pyplot as plt

    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()

    # Columns to fill with 'missing' and 0 respectively
    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    # Fill NaN values with 'missing' for the specified columns
    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    
    # Fill NaN values with 0 for the specified columns
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Convert specified columns to category type
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define preprocessor
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor and transform the training and testing data
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if GridSearch:
        # Define the model
        regressor = KNeighborsRegressor()

        # Define parameter grid for GridSearch
        param_grid = {
            'n_neighbors': [2, 3, 5, 7]
        }

        # Perform GridSearch with cross-validation
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train_transformed, y_train)

        # Print best parameters
        print("Best parameters found by GridSearch:", grid_search.best_params_)

        # Use best estimator from grid search
        regressor = grid_search.best_estimator_
    else:
        # Define the model with default parameters
        regressor = KNeighborsRegressor(n_neighbors=2)

        # Train the model
        regressor.fit(X_train_transformed, y_train)

    # Predict on test data
    y_pred = regressor.predict(X_test_transformed)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)
    print("Mean Absolute Error:", mae)
    
    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)
    
    return imputed_df





def nn_model(training_df, scaler, epochs_number, batch_size, df_estimeres, GridSearch=True):
    """
    Trains a neural network model for predicting new_oms values with an optional GridSearch for hyperparameter tuning.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    epochs_number (int): Number of epochs for training the neural network.
    batch_size (int): Batch size for training the neural network.
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from scikeras.wrappers import KerasRegressor
    import tensorflow as tf
    import matplotlib.pyplot as plt

    def build_nn_model(input_shape, learning_rate=0.001, dropout_rate=0.5, neurons_layer1=64, neurons_layer2=32, activation='relu', optimizer='adam'):
        """
        Builds and compiles a neural network model.

        Parameters:
        input_shape (int): Number of input features.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate for regularization.
        neurons_layer1 (int): Number of neurons in the first layer.
        neurons_layer2 (int): Number of neurons in the second layer.
        activation (str): Activation function to use.
        optimizer (str): Optimizer to use.

        Returns:
        tf.keras.Model: Compiled neural network model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(neurons_layer1, input_shape=(input_shape,), activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(neurons_layer2, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
        return model

    # Prepare the data
    df = training_df.copy()
    imputed_df = df_estimeres.copy()

    # Fill NaN values in specified columns
    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Convert categorical columns to category type
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define preprocessor
    categorical_features = categorical_columns
    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    input_shape = X_train_transformed.shape[1]

    # Wrap the model with KerasRegressor
    nn_model = KerasRegressor(build_fn=build_nn_model, input_shape=input_shape, epochs=epochs_number, batch_size=batch_size, verbose=0)

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    if GridSearch:
        # Perform Grid Search for hyperparameter tuning
        param_grid = {
            'epochs': [100, 200, 300, 400],
            'batch_size': [10, 32, 64, 128],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.3, 0.5, 0.7],
            'neurons_layer1': [32, 64, 128],
            'neurons_layer2': [16, 32, 64],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam', 'sgd', 'rmsprop']
        }
        grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train_transformed, y_train, callbacks=[early_stopping])
        print("Best parameters found by GridSearch:", grid_search.best_params_)
        nn_model = grid_search.best_estimator_
    else:
        # Train the model with provided parameters
        nn_model.fit(X_train_transformed, y_train, callbacks=[early_stopping])

    # Predict on test data
    y_pred = nn_model.predict(X_test_transformed).flatten()  # Ensure y_pred is 1-dimensional

    # Set negative predictions to zero
    y_pred = np.maximum(y_pred, 0)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Median Absolute Error (MedAE): {medae}")
    print(f"R-squared score: {r_squared}")

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = nn_model.predict(imputed_X_transformed)
    
    return imputed_df




def xgboost_n3_klass(df):
    """
    Trains an XGBoost classifier to predict 'n3' categories with preprocessing for numerical and categorical data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'n3' as the target variable.

    Returns:
    None
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt

    # Filter out sparse classes
    min_samples_per_class = 50
    value_counts = df['n3'].value_counts()
    to_remove = value_counts[value_counts < min_samples_per_class].index
    df = df[~df['n3'].isin(to_remove)]

    # Convert target to integer labels and store the mapping
    labels, unique = pd.factorize(df['n3'])
    df['n3_encoded'] = labels
    n3_mapping = dict(zip(labels, unique))

    # Identify categorical and numerical columns excluding the target
    non_feature_cols = ['n3']  # 'n3' is now not a feature
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in non_feature_cols]
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    numerical_cols.remove('n3_encoded')  # Assume 'n3_encoded' is the new target variable

    # Preprocessing for numerical data: simple imputer with median strategy
    numerical_transformer = SimpleImputer(strategy='median')

    # Preprocessing for categorical data: impute missing values and apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the XGBoost classifier
    model = xgb.XGBClassifier(objective='multi:softprob', random_state=42, eval_metric='mlogloss')

    # Create a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Prepare data
    X = df.drop(non_feature_cols + ['n3_encoded'], axis=1)
    y = df['n3_encoded']  # Correct target variable

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    # Generate a mapping from labels to original 'n3' categories if not done previously
    n3_mapping = {idx: label for idx, label in enumerate(pd.unique(df['n3']))}

    # Safely convert predictions and true values back to original labels using the mapping
    y_pred_labels = [n3_mapping.get(label, 'Unknown') for label in y_pred]
    y_test_labels = [n3_mapping.get(label, 'Unknown') for label in y_test]

    # Print accuracy and classification report using the original 'n3' labels
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Plot learning curve
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters:
        estimator: object type that implements the "fit" and "predict" methods
        title: string, Title for the chart
        X: array-like, shape (n_samples, n_features), Training vector
        y: array-like, shape (n_samples) or (n_samples, n_features), Target relative to X for classification or regression
        ylim: tuple, shape (ymin, ymax), optional, Defines minimum and maximum y-values plotted
        cv: int, cross-validation generator or an iterable, optional, Determines the cross-validation splitting strategy
        n_jobs: int or None, optional, Number of jobs to run in parallel
        train_sizes: array-like, shape (n_ticks,), Relative or absolute numbers of training examples that will be used to generate the learning curve

        Returns:
        plt: Matplotlib plot object
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        return plt

    plot_learning_curve(clf, "Learning Curve for XGBoost Classifier", X_train, y_train, cv=5)
    plt.show()




def knn_n3_klass(df):
    """
    Trains a K-Nearest Neighbors (KNN) classifier to predict 'n3' categories with preprocessing 
    for numerical and categorical data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'n3' as the target variable.

    Returns:
    None
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt

    # Filter out sparse classes
    min_samples_per_class = 100
    value_counts = df['n3'].value_counts()
    to_remove = value_counts[value_counts < min_samples_per_class].index
    df = df[~df['n3'].isin(to_remove)]

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Specify columns to exclude from features
    non_feature_cols = ['n3']
    categorical_cols = [col for col in categorical_cols if col not in non_feature_cols]

    # Preprocessing for numerical data: impute missing values and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Scale features
    ])

    # Preprocessing for categorical data: impute missing values and apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the KNN classifier
    model = KNeighborsClassifier(n_neighbors=5)

    # Create a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Prepare data
    X = df.drop(non_feature_cols, axis=1)
    y = df['n3']

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Plot learning curve
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters:
        estimator: object type that implements the "fit" and "predict" methods
        title: string, Title for the chart
        X: array-like, shape (n_samples, n_features), Training vector
        y: array-like, shape (n_samples) or (n_samples, n_features), Target relative to X for classification or regression
        ylim: tuple, shape (ymin, ymax), optional, Defines minimum and maximum y-values plotted
        cv: int, cross-validation generator or an iterable, optional, Determines the cross-validation splitting strategy
        n_jobs: int or None, optional, Number of jobs to run in parallel
        train_sizes: array-like, shape (n_ticks,), Relative or absolute numbers of training examples that will be used to generate the learning curve

        Returns:
        plt: Matplotlib plot object
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        return plt

    plot_learning_curve(clf, "Learning Curve for KNN Classifier", X_train, y_train, cv=5)
    plt.show()

    

def test_results(df, aar):
    
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{aar}/statistikkfil_bedrifter_nr.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    bedrift_2 = table.to_pandas()

    # change pd option to show all columns
    pd.set_option("display.max_columns", None)

    bedrift_2 = bedrift_2[['orgnr_bedrift', 'omsetning', 'nopost_driftskostnader']]
    
    bedrift_1 = df[['v_orgnr', 'oms', 'new_drkost', 'regtype']]

    # rename 
    bedrift_1.rename(columns={"v_orgnr": "orgnr_bedrift"}, inplace=True)
    print(bedrift_1.shape)
    test = bedrift_1.merge(bedrift_2, on='orgnr_bedrift', how='left')
    test = test.drop_duplicates()
    test = test.dropna()
    
    # Calculate the absolute difference
    test['oms_diff'] = (test['oms'] - test['omsetning']).abs()

    # Sort the DataFrame by the 'oms_diff' column in descending order
    test_sorted = test.sort_values(by='oms_diff', ascending=False)

    # Display the sorted DataFrame
    test_sorted.head()

    # create new df where regtype == 02

    test_02 = test_sorted[test_sorted['regtype'] == '02']
    
    # Assuming your DataFrame is named 'test' and has columns 'oms' and 'omsetning'
    oms = test['oms']
    omsetning = test['omsetning']

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(omsetning, oms)
    print(f'Mean Absolute Error for entire delreg: {mae}')

    # Calculate R-squared (R²)
    r2 = r2_score(omsetning, oms)
    print(f'R² Score for entire delreg: {r2}')
    
    
    print(f'-----------------------------------')
    
    # Assuming your DataFrame is named 'test' and has columns 'oms' and 'omsetning'
    oms = test_02['oms']
    omsetning = test_02['omsetning']

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(omsetning, oms)
    print(f'Mean Absolute Error for reg_type 02: {mae}')

    # Calculate R-squared (R²)
    r2 = r2_score(omsetning, oms)
    print(f'R² Score for reg_type 02: {r2}')


def fetch_foretak_data(aar):
    
    # Fetch paths to all Parquet files for the specified year related to foretak (enterprises)
    fil_path = [
        f for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{aar}/statistikkfil_foretak_pub.parquet"
        ) if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple Parquet files into a single Arrow Table
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert the Arrow Table into a Pandas DataFrame
    foretak_pub = table.to_pandas()

    # Create a new column 'n3' extracting the first four characters from 'naring_f' column
    # Create a new column 'n2' extracting the first two characters from 'naring_f' column
    foretak_pub['n3'] = foretak_pub['naring_f'].str[:4]
    foretak_pub['n2'] = foretak_pub['naring_f'].str[:2]

    # Filter data where 'n2' indicates specific industry codes relevant to the analysis
    foretak_varendel = foretak_pub[(foretak_pub['n2'] == '45') | (foretak_pub['n2'] == '46') | (foretak_pub['n2'] == '47')]

    # Select only the relevant columns for further processing
    foretak_varendel = foretak_varendel[['orgnr_foretak', 'naring_f', 'n2', 'n3', 'bearbeidingsverdi',
                                         'produktinnsats', 'produksjonsverdi', 'omsetning', 
                                         'nopost_driftsresultat', 'nopost_driftskostnader',
                                         'nopost_driftsinntekter', 'sysselsetting_syss']]
    
    return foretak_pub, foretak_varendel
