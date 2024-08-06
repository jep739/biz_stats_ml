import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
import getpass
import os
import xgboost as xgb
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
# import sgis as sg
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from pyjstat import pyjstat
# import plotly.express as px
from ipywidgets import interact, Dropdown
# from klass import search_classification

import sys

sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
import kommune_translate
import kommune
import ml_modeller

fs = FileClient.get_gcs_file_system()
import numpy as np


import warnings

warnings.filterwarnings("ignore")

import math

# good_df = ao.rette_bedrifter(good_df)

import input_data
# import create_datafiles

from joblib import Parallel, delayed
import multiprocessing

import time
import create_datafiles





def create_bedrift_fil(year, model, rate, scaler, GridSearch=False):

    start_time = time.time()

    # Generate the data required for processing using the create_datafiles.main function
    current_year_good_oms, current_year_bad_oms, v_orgnr_list_for_imputering, training_data, imputatable_df, time_series_df = create_datafiles.main(year, rate)

    # Construct the function name dynamically based on the model parameter
    function_name = f"{model}"

    # Use getattr to dynamically retrieve the function from the ml_modeller module
    function_to_call = getattr(ml_modeller, function_name)

    # Call the retrieved function with the training data, scaler, imputatable DataFrame, and GridSearch parameter
    # Check if the model is 'nn_model' and set additional parameters if true
    if model == 'nn_model':
        epochs_number = 200
        batch_size = 500
        # Call the function with the additional parameters
        imputed_df = function_to_call(training_data, scaler, epochs_number, batch_size, imputatable_df, GridSearch=GridSearch)
    else:
        # Call the function without the additional parameters
        imputed_df = function_to_call(training_data, scaler, imputatable_df, GridSearch=GridSearch)

    # Extract the relevant columns from the imputed DataFrame for merging
    df_to_merge = imputed_df[['v_orgnr', 'year', 'id', 'predicted_oms']]

    # Merge the imputed DataFrame with the current year's bad data on 'v_orgnr', 'id', and 'year'
    bad_df = pd.merge(current_year_bad_oms, df_to_merge, on=['v_orgnr', 'id', 'year'], how='left')

    # Assign the 'predicted_oms' values to a new column 'new_oms'
    bad_df['new_oms'] = bad_df['predicted_oms']

    # Drop the 'predicted_oms' column as it is no longer needed
    bad_df.drop(['predicted_oms'], axis=1, inplace=True)

    # Concatenate the good data and the modified bad data into a single DataFrame
    good_df = pd.concat([current_year_good_oms, bad_df], ignore_index=True)

    # Filter the DataFrame to include only rows where 'lopenr' equals 1
    good_df = good_df[good_df['lopenr'] == 1]

    # Ensure that 'new_oms' values are non-negative
    good_df['new_oms'] = good_df['new_oms'].apply(lambda x: 0 if x < 0 else x)

    # Drop the 'tot_oms_fordelt' column as it will be recalculated
    good_df.drop(['tot_oms_fordelt'], axis=1, inplace=True)

    # Group by 'id' and calculate the sum of 'new_oms' for each group
    grouped = good_df.groupby("id")[["new_oms"]].sum().reset_index()

    # Rename the 'new_oms' column to 'tot_oms_fordelt' in the grouped DataFrame
    grouped.rename(columns={"new_oms": "tot_oms_fordelt"}, inplace=True)

    # Merge the grouped DataFrame back into the original DataFrame on 'id'
    good_df = pd.merge(good_df, grouped, on="id", how="left")

    # Convert necessary columns to appropriate data types
    good_df['id'] = good_df['id'].astype(str)
    good_df['nacef_5'] = good_df['nacef_5'].astype(str)
    good_df['orgnr_n_1'] = good_df['orgnr_n_1'].astype(str)
    good_df['b_kommunenr'] = good_df['b_kommunenr'].astype(str)
    good_df['forbruk'] = good_df['forbruk'].astype(float)
    good_df['salgsint'] = good_df['salgsint'].astype(float)
    good_df['tmp_no_p4005'] = good_df['tmp_no_p4005'].astype(float)

    
   
    # Define a condition to identify rows where either 'foretak_omsetning' or 'foretak_driftskostnad' is 0
    condition = (good_df['foretak_omsetning'] == 0) | (good_df['foretak_driftskostnad'] == 0)

    # Drop the rows that meet the defined condition
    good_df = good_df[~condition]

    # Convert 'gjeldende_driftsk_kr' to numeric, coercing errors to NaN
    good_df["gjeldende_driftsk_kr"] = pd.to_numeric(good_df["gjeldende_driftsk_kr"], errors="coerce")

    # Convert 'b_sysselsetting_syss' to numeric, coercing errors to NaN
    good_df["b_sysselsetting_syss"] = pd.to_numeric(good_df["b_sysselsetting_syss"], errors="coerce")

    # Drop the 'tot_driftskost_fordelt' column as it will be recalculated
    good_df.drop(['tot_driftskost_fordelt'], axis=1, inplace=True)

    # Create a new column 'driftsk' which is a copy of 'gjeldende_driftsk_kr'
    good_df["driftsk"] = good_df["gjeldende_driftsk_kr"]

    # Group by 'id' and calculate the sum of 'driftsk' for each group
    grouped = good_df.groupby("id")[["driftsk"]].sum().reset_index()

    # Rename the 'driftsk' column to 'tot_driftskost_fordelt' in the grouped DataFrame
    grouped.rename(columns={"driftsk": "tot_driftskost_fordelt"}, inplace=True)

    # Merge the grouped DataFrame back into the original DataFrame on 'id'
    good_df = pd.merge(good_df, grouped, on="id", how="left")

    # Convert 'tot_driftskost_fordelt' to numeric, coercing errors to NaN
    good_df["tot_driftskost_fordelt"] = pd.to_numeric(good_df["tot_driftskost_fordelt"], errors="coerce")

    # Convert 'driftsk' to numeric, coercing errors to NaN
    good_df["driftsk"] = pd.to_numeric(good_df["driftsk"], errors="coerce")

    # Ensure 'b_sysselsetting_syss' is numeric, coercing errors to NaN
    good_df["b_sysselsetting_syss"] = pd.to_numeric(good_df["b_sysselsetting_syss"], errors="coerce")

    # Convert 'lonn' to numeric, replacing commas with dots and handling errors
    good_df["lonn"] = good_df["lonn"].str.replace(',', '.').astype(float)

    # Scale down 'lonn' values by dividing by 100
    good_df["lonn"] = good_df["lonn"] / 100

    # Calculate 'drkost_share' based on certain conditions
    good_df["drkost_share"] = good_df.apply(
        lambda row: row["lonn"] if row["foretak_driftskostnad"] != 0 and row["tot_driftskost_fordelt"] == 0 else (row["driftsk"] / row["tot_driftskost_fordelt"] if row["tot_driftskost_fordelt"] != 0 else np.nan),
        axis=1
    )

    # Replace infinite values with NaN in 'drkost_share'
    good_df['drkost_share'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values in 'drkost_share' with 0
    good_df['drkost_share'].fillna(0, inplace=True)

    # Calculate the total 'b_sysselsetting_syss' per 'id'
    good_df['total_syss'] = good_df.groupby('id')['b_sysselsetting_syss'].transform('sum')

    # Ensure 'total_syss' is numeric, coercing errors to NaN
    good_df["total_syss"] = pd.to_numeric(good_df["total_syss"], errors="coerce")

    # Calculate the share of 'b_sysselsetting_syss' per 'id'
    good_df['syss_share'] = good_df['b_sysselsetting_syss'] / good_df['total_syss']

    # Update 'drkost_share' for specific conditions
    good_df.loc[
        (good_df['tot_driftskost_fordelt'] == 0) & 
        (good_df['drkost_share'] == 0) & 
        (good_df['foretak_driftskostnad'] != 0), 
        'drkost_share'
    ] = good_df['syss_share']

    # Round 'drkost_share' to 10 decimal points
    good_df['drkost_share'] = good_df['drkost_share'].round(10)

    # Calculate 'new_drkost' by multiplying 'drkost_share' with 'foretak_driftskostnad'
    good_df["new_drkost"] = good_df["drkost_share"] * good_df["foretak_driftskostnad"]

    # Replace NaN values in 'new_drkost' with the values from 'gjeldende_driftsk_kr'
    good_df["new_drkost"].fillna(good_df["gjeldende_driftsk_kr"], inplace=True)

    # Ensure 'new_drkost' is of type float
    good_df['new_drkost'] = good_df['new_drkost'].astype(float)

    # Set 'new_drkost' to 0 where 'foretak_driftskostnad' is 0
    good_df.loc[good_df['foretak_driftskostnad'] == 0, 'new_drkost'] = 0

    # Set 'new_oms' to 0 where 'foretak_omsetning' is 0
    good_df.loc[good_df['foretak_omsetning'] == 0, 'new_oms'] = 0

    # Ensure 'new_drkost' is of type float (redundant, but keeping for consistency)
    good_df['new_drkost'] = good_df['new_drkost'].astype(float)

    # Drop the 'drkost_share' column as it is no longer needed
    good_df.drop(['drkost_share'], axis=1, inplace=True)

    # Drop the 'tot_oms_fordelt' column as it will be recalculated
    good_df.drop(['tot_oms_fordelt'], axis=1, inplace=True)

    # Group by 'id' and calculate the sum of 'new_oms' for each group
    grouped = good_df.groupby("id")[["new_oms"]].sum().reset_index()

    # Rename the 'new_oms' column to 'tot_oms_fordelt' in the grouped DataFrame
    grouped.rename(columns={"new_oms": "tot_oms_fordelt"}, inplace=True)

    # Merge the grouped DataFrame back into the original DataFrame on 'id'
    good_df = pd.merge(good_df, grouped, on="id", how="left")

    # Create boolean masks for rows where 'regtype' is '04' and not '04'
    mask_regtype_04 = good_df['regtype'] == '04'
    mask_regtype_not_04 = good_df['regtype'] != '04'

    # Update 'new_oms' to be equal to 'new_drkost' where 'regtype' is '04'
    good_df.loc[mask_regtype_04, 'new_oms'] = good_df.loc[mask_regtype_04, 'new_drkost']

    # Group by 'id' and sum 'new_oms' for rows where 'regtype' is '04'
    total_helper_oms = good_df.loc[mask_regtype_04].groupby('id')['new_oms'].sum().reset_index()

    # Rename the aggregated 'new_oms' to 'new_oms_total_helper' for clarity
    total_helper_oms.rename(columns={'new_oms': 'new_oms_total_helper'}, inplace=True)

    # Merge the aggregated result back into the original DataFrame
    good_df = pd.merge(good_df, total_helper_oms, on='id', how='left', suffixes=('', '_total_helper'))

    # Reset index to ensure it's unique
    good_df.reset_index(drop=True, inplace=True)

    # Convert 'foretak_omsetning' to numeric, setting errors to NaN
    good_df['foretak_omsetning'] = pd.to_numeric(good_df['foretak_omsetning'], errors='coerce')

    # Fill NaN values that might have been introduced by conversion errors
    good_df['foretak_omsetning'].fillna(0, inplace=True)
    good_df['new_oms_total_helper'].fillna(0, inplace=True)

    # Ensure 'new_oms_total_helper' is of type float
    good_df['new_oms_total_helper'] = pd.to_numeric(good_df['new_oms_total_helper'], errors='coerce')

    # Calculate 'total_rest_oms' by subtracting 'new_oms_total_helper' from 'foretak_omsetning'
    good_df['total_rest_oms'] = good_df['foretak_omsetning'] - good_df['new_oms_total_helper']

    # Group by 'id' and sum 'new_oms' for rows where 'regtype' is not '04'
    total_non_helper_oms = good_df.loc[mask_regtype_not_04].groupby('id')['new_oms'].sum().reset_index()

    # Merge the aggregated result back into the original DataFrame
    good_df = pd.merge(good_df, total_non_helper_oms, on='id', how='left', suffixes=('', '_total_non_helper'))

    # Fill NaN values in 'new_oms_total_non_helper' with 0
    good_df['new_oms_total_non_helper'].fillna(0, inplace=True)

    # Convert 'new_oms' and 'new_oms_total_non_helper' to numeric, coercing errors to NaN
    good_df['new_oms'] = pd.to_numeric(good_df['new_oms'], errors='coerce')
    good_df['new_oms_total_non_helper'] = pd.to_numeric(good_df['new_oms_total_non_helper'], errors='coerce')

    # Convert NaN values in 'new_oms_total_non_helper' to 0
    good_df['new_oms_total_non_helper'].fillna(0, inplace=True)

    # Calculate total 'lonn' per 'id' excluding rows where 'regtype' is '04'
    good_df['total_lonn_non_04'] = good_df[mask_regtype_not_04].groupby('id')['lonn'].transform('sum')

    # Ensure 'total_lonn_non_04' is numeric and handle any conversion issues
    good_df['total_lonn_non_04'] = pd.to_numeric(good_df['total_lonn_non_04'], errors='coerce')

    # Recalculate 'lonn' for non-'04' rows to sum to 100% per 'id'
    good_df['lonn_non_04_share'] = np.where(
        mask_regtype_not_04,
        good_df['lonn'] / good_df['total_lonn_non_04'],
        0
    )

    # Fill NaN values in 'lonn_non_04_share' with 0
    good_df['lonn_non_04_share'].fillna(0, inplace=True)

    # Calculate total 'b_sysselsetting_syss' per 'id' excluding rows where 'regtype' is '04'
    good_df['total_syss_non_04'] = good_df[mask_regtype_not_04].groupby('id')['b_sysselsetting_syss'].transform('sum')

    # Calculate the share of 'b_sysselsetting_syss' per 'id' excluding rows where 'regtype' is '04'
    good_df['syss_share_non_04'] = good_df['b_sysselsetting_syss'] / good_df['total_syss_non_04']

    # Fill NaN values in 'syss_share_non_04' with 0
    good_df['syss_share_non_04'].fillna(0, inplace=True)

    # Calculate 'oms_share_non_helpers' based on the given conditions
    good_df['oms_share_non_helpers'] = good_df.apply(
        lambda row: 1 if row['total_rest_oms'] == 0 else (
            row['lonn_non_04_share'] if row['foretak_omsetning'] != 0 and row['tot_oms_fordelt'] == 0 else (
                row["new_oms"] / row["new_oms_total_non_helper"] if row["new_oms_total_non_helper"] != 0 else row['syss_share_non_04'])),
        axis=1
    )


    ######################################

    # Handle any NaN or inf values in 'oms_share_non_helpers'
    good_df['oms_share_non_helpers'].replace([np.inf, -np.inf], np.nan, inplace=True)
    good_df['oms_share_non_helpers'].fillna(0, inplace=True)

    # Ensure 'oms_share_non_helpers' is set to 0 where 'regtype' is '04'
    good_df.loc[good_df['regtype'] == '04', 'oms_share_non_helpers'] = 0

    # Update 'new_oms' with the calculated 'oms_share_non_helpers'
    good_df['new_oms'] = np.where(good_df['regtype'] == '04', 
                                  good_df['new_oms'], 
                                  good_df['oms_share_non_helpers'] * good_df['total_rest_oms'])

    # Define the values for 'w_naring_vh', 'w_nace1_ikke_vh', and 'w_nace2_ikke_vh'
    w_naring_vh = ("45", "46", "47")
    w_nace1_ikke_vh = "45.403"
    w_nace2_ikke_vh = ("45.2", "46.1")

    # Create a copy of 'good_df' to 'enhetene_brukes'
    enhetene_brukes = good_df.copy()

    # Delete 'good_df' to free up memory
    del good_df

    # Initialize 'vhbed' column to 0
    enhetene_brukes["vhbed"] = 0

    # Set 'vhbed' to 1 if the first two characters of 'tmp_sn2007_5' are in 'w_naring_vh'
    enhetene_brukes.loc[
        enhetene_brukes["tmp_sn2007_5"].str[:2].isin(w_naring_vh), "vhbed"
    ] = 1

    # Set 'vhbed' to 0 if 'tmp_sn2007_5' equals 'w_nace1_ikke_vh'
    enhetene_brukes.loc[enhetene_brukes["tmp_sn2007_5"] == w_nace1_ikke_vh, "vhbed"] = 0

    # Set 'vhbed' to 0 if the first four characters of 'tmp_sn2007_5' are in 'w_nace2_ikke_vh'
    enhetene_brukes.loc[
        enhetene_brukes["tmp_sn2007_5"].str[:4].isin(w_nace2_ikke_vh), "vhbed"
    ] = 0

    # Drop duplicate rows based on the specified subset of columns
    enhetene_brukes = enhetene_brukes.drop_duplicates(subset=["orgnr_n_1", "lopenr", "radnr", "v_orgnr"])

    # Calculate 'check' as the difference between 'foretak_driftskostnad' and 'forbruk'
    enhetene_brukes['check'] = enhetene_brukes["foretak_driftskostnad"] - enhetene_brukes["forbruk"]

    # Adjust 'forbruk' based on 'check'
    enhetene_brukes['forbruk'] = np.where(
        enhetene_brukes['check'] < 0, enhetene_brukes['forbruk'] / 1000, enhetene_brukes['forbruk']
    )

    # Select relevant columns for 'salgsint_forbruk'
    salgsint_forbruk = enhetene_brukes[
        [
            "orgnr_n_1",
            "lopenr",
            "v_orgnr",
            "forbruk",
            "salgsint",
            "radnr",
            "nacef_5",
            "tmp_sn2007_5",
            "new_oms",
            "vhbed",
        ]
    ]

    # Identify unique 'orgnr_n_1' where 'vhbed' is True
    har = salgsint_forbruk[salgsint_forbruk.groupby("orgnr_n_1")["vhbed"].transform("any")]
    har = har[["orgnr_n_1"]]
    har.drop_duplicates(inplace=True)

    # Identify unique 'orgnr_n_1' where 'vhbed' is False
    ikke_har = salgsint_forbruk[~salgsint_forbruk.groupby("orgnr_n_1")["vhbed"].transform("any")]
    ikke_har = ikke_har[["orgnr_n_1"]]
    ikke_har.drop_duplicates(inplace=True)

    # Add a column 'ikkevbed' with value 1 to 'ikke_har'
    ikke_har["ikkevbed"] = 1

    # Merge ikke_har into salgsint_forbruk with a left join on the 'orgnr_n_1' column
    salgsint_forbruk_update1 = pd.merge(
        salgsint_forbruk, ikke_har, on="orgnr_n_1", how="left"
    )

    # Uncomment if you want to fill NaN values in 'ikkevbed' with 0
    # salgsint_forbruk_update1['ikkevbed'].fillna(0, inplace=True)

    # Update 'vhbed' to 1 where 'ikkevbed' is 1
    salgsint_forbruk_update1.loc[salgsint_forbruk_update1["ikkevbed"] == 1, "vhbed"] = 1

    # Create sum1 DataFrame for vhbed=1
    sum1 = (
        salgsint_forbruk_update1[salgsint_forbruk_update1["vhbed"] == 1]
        .groupby(["orgnr_n_1", "lopenr"])["new_oms"]
        .sum()
        .reset_index()
    )
    sum1.rename(columns={"new_oms": "sumoms_vh"}, inplace=True)

    # Create sum2 DataFrame for vhbed=0
    sum2 = (
        salgsint_forbruk_update1[salgsint_forbruk_update1["vhbed"] == 0]
        .groupby(["orgnr_n_1", "lopenr"])["new_oms"]
        .sum()
        .reset_index()
    )
    sum2.rename(columns={"new_oms": "sumoms_andre"}, inplace=True)

    # Merge sum1 and sum2
    sum3 = pd.merge(sum1, sum2, on=["orgnr_n_1", "lopenr"], how="outer")

    # Merge sum3 with salgsint_forbruk_update1
    salgsint_forbruk_update2 = pd.merge(
        salgsint_forbruk_update1, sum3, on=["orgnr_n_1", "lopenr"], how="outer"
    )

    # Sort the DataFrame by 'orgnr_n_1', 'lopenr', and 'radnr'
    salgsint_forbruk_update2.sort_values(by=["orgnr_n_1", "lopenr", "radnr"], inplace=True)
    salgsint_forbruk_update2.sort_values(by=["orgnr_n_1", "lopenr", "vhbed"], inplace=True)

    # Copy the DataFrame for further updates
    salgsint_forbruk_update3 = salgsint_forbruk_update2.copy()

    # Sort the DataFrame by 'orgnr_n_1' and 'lopenr'
    salgsint_forbruk_update3.sort_values(by=["orgnr_n_1", "lopenr"], inplace=True)

    # Create a new variable 'vhf' based on the values of 'vhbed'
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3.groupby(
        ["orgnr_n_1", "lopenr"]
    )["vhbed"].transform("first")

    # Retain the value of 'vhf' from the first observation in each group
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3.groupby(
        ["orgnr_n_1", "lopenr"]
    )["vhf"].transform("first")

    # Apply labels to the variables
    salgsint_forbruk_update3["vhbed"] = salgsint_forbruk_update3["vhbed"].astype(str)
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3["vhf"].astype(str)

    label_map_vhbed = {"1": "varehandelsbedrift", "0": "annen type bedrift"}
    label_map_vhf = {
        "1": "foretaket har kun varehandelsbedrifter eller ingen",
        "0": "har varehandel og annen bedrift (blandingsnæringer)",
    }

    salgsint_forbruk_update3["vhbed"] = salgsint_forbruk_update3["vhbed"].map(
        label_map_vhbed
    )
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3["vhf"].map(label_map_vhf)

    # Filter rows where vhf is 'foretaket har kun varehandelsbedrifter eller ingen'
    vhf_condition = (
        salgsint_forbruk_update3["vhf"]
        == "foretaket har kun varehandelsbedrifter eller ingen"
    )
    vhf_df = salgsint_forbruk_update3.loc[vhf_condition]


    # Filter rows where vhf is not 'foretaket har kun varehandelsbedrifter eller ingen'
    andre_df = salgsint_forbruk_update3.loc[~vhf_condition]

    # Calculate 'nokkel' for 'vhf_df'
    vhf_df["nokkel"] = vhf_df["new_oms"] / vhf_df["sumoms_vh"]

    # Convert 'salgsint' and 'forbruk' columns to numeric
    vhf_df["salgsint"] = pd.to_numeric(vhf_df["salgsint"], errors="coerce")
    vhf_df["forbruk"] = pd.to_numeric(vhf_df["forbruk"], errors="coerce")

    # Calculate 'bedr_salgsint' and 'bedr_forbruk' for 'vhf_df'
    vhf_df["bedr_salgsint"] = round(vhf_df["salgsint"] * vhf_df["nokkel"])
    vhf_df["bedr_forbruk"] = round(vhf_df["forbruk"] * vhf_df["nokkel"])

    # Convert 'forbruk' and 'salgsint' columns to numeric in 'andre_df'
    andre_df["forbruk"] = pd.to_numeric(andre_df["forbruk"], errors="coerce")
    andre_df["salgsint"] = pd.to_numeric(andre_df["salgsint"], errors="coerce")

    # Calculate 'avanse' for 'andre_df'
    andre_df["avanse"] = andre_df["forbruk"] / andre_df["salgsint"]

    # Filter rows where vhbed is 'varehandelsbedrift'
    vh_bedriftene = andre_df[andre_df["vhbed"] == "varehandelsbedrift"].copy()

    # Calculate 'nokkel', 'bedr_salgsint', and 'bedr_forbruk' for vh-bedriftene
    vh_bedriftene["nokkel"] = vh_bedriftene["new_oms"] / vh_bedriftene["sumoms_vh"]
    vh_bedriftene["bedr_salgsint"] = round(vh_bedriftene["salgsint"] * vh_bedriftene["nokkel"])
    vh_bedriftene.loc[
        vh_bedriftene["bedr_salgsint"] > vh_bedriftene["new_oms"], "bedr_salgsint"
    ] = vh_bedriftene["new_oms"]
    vh_bedriftene["bedr_forbruk"] = round(
        vh_bedriftene["bedr_salgsint"] * vh_bedriftene["avanse"]
    )

    # Summarize vh-bedriftene
    brukt1 = (
        vh_bedriftene.groupby(["orgnr_n_1", "lopenr"])
        .agg({"bedr_salgsint": "sum", "bedr_forbruk": "sum"})
        .reset_index()
    )

    # Merge summarized values back to 'andre'
    andre = pd.merge(andre_df, brukt1, on=["orgnr_n_1", "lopenr"], how="left")

    # Calculate 'resten1' and 'resten2'
    andre["resten1"] = andre["salgsint"] - andre["bedr_salgsint"]
    andre["resten2"] = andre["forbruk"] - andre["bedr_forbruk"]

    # Filter rows where vhbed is not 'varehandelsbedrift'
    blanding_av_vh_og_andre = andre[andre["vhbed"] != "varehandelsbedrift"].copy()

    # Calculate 'nokkel', 'bedr_salgsint', and 'bedr_forbruk' for blending of vh and other industries
    blanding_av_vh_og_andre["nokkel"] = (
        blanding_av_vh_og_andre["new_oms"] / blanding_av_vh_og_andre["sumoms_andre"]
    )
    blanding_av_vh_og_andre["bedr_salgsint"] = round(
        blanding_av_vh_og_andre["resten1"] * blanding_av_vh_og_andre["nokkel"]
    )
    blanding_av_vh_og_andre["bedr_forbruk"] = round(
        blanding_av_vh_og_andre["resten2"] * blanding_av_vh_og_andre["nokkel"]
    )

    # Combine the two subsets back into 'andre'
    andre = pd.concat([vh_bedriftene, blanding_av_vh_og_andre], ignore_index=True)

    # Sort the combined DataFrame by 'orgnr_n_1' and 'lopenr'
    andre.sort_values(by=["orgnr_n_1", "lopenr"], inplace=True)

    # Combine 'vhf_df' and 'andre' into 'oppdatere_hv'
    oppdatere_hv = pd.concat([vhf_df, andre], ignore_index=True)

    # Select specific columns for 'oppdatere_hv'
    oppdatere_hv = oppdatere_hv[
        ["orgnr_n_1", "lopenr", "radnr", "bedr_forbruk", "bedr_salgsint"]
    ]

    # Merge 'oppdatere_hv' with 'enhetene_brukes'
    enhetene_brukes2 = pd.merge(
        enhetene_brukes, oppdatere_hv, on=["orgnr_n_1", "lopenr", "radnr"]
    )

    # Identify IDs that appear more than once
    duplicate_ids = enhetene_brukes2['id'][enhetene_brukes2['id'].duplicated(keep=False)]

    # Update 'regtype' to '02' where 'regtype' is '01' and the ID appears more than once
    enhetene_brukes2.loc[(enhetene_brukes2['regtype'] == '01') & (enhetene_brukes2['id'].isin(duplicate_ids)), 'regtype'] = '02'

    # Update specific columns where 'regtype' is '01'
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'new_oms'] = enhetene_brukes2['foretak_omsetning']
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'new_drkost'] = enhetene_brukes2['foretak_driftskostnad']
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'bedr_salgsint'] = enhetene_brukes2['salgsint']
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'bedr_forbruk'] = enhetene_brukes2['forbruk']

    # Create a copy of 'enhetene_brukes2'
    rettes2 = enhetene_brukes2.copy()

    # Delete 'enhetene_brukes2' to free up memory
    del enhetene_brukes2

    # Update 'oms' and 'driftsk' in 'rettes2'
    rettes2["oms"] = rettes2["new_oms"]
    rettes2["driftsk"] = rettes2["gjeldende_driftsk_kr"]

    # Convert columns to numeric
    rettes2["tot_driftskost_fordelt"] = pd.to_numeric(rettes2["tot_driftskost_fordelt"], errors="coerce")
    rettes2["driftsk"] = pd.to_numeric(rettes2["driftsk"], errors="coerce")

    # Fill NaN values in 'new_drkost' with 0
    rettes2["new_drkost"] = rettes2["new_drkost"].fillna(0)

    # Assign 'new_drkost' to 'drkost_temp'
    rettes2["drkost_temp"] = rettes2["new_drkost"]

    # Fill NaN values in 'drkost_temp' with 0
    rettes2["drkost_temp"] = rettes2["drkost_temp"].fillna(0)

    # Replace commas with dots in 'gjeldende_lonn_kr' and convert to numeric
    rettes2["gjeldende_lonn_kr"] = rettes2["gjeldende_lonn_kr"].str.replace(",", ".")
    rettes2["gjeldende_lonn_kr"] = pd.to_numeric(rettes2["gjeldende_lonn_kr"], errors="coerce").fillna(0)

    # Convert 'bedr_forbruk' to numeric and fill NaN values with 0
    rettes2["bedr_forbruk"] = pd.to_numeric(rettes2["bedr_forbruk"], errors="coerce").fillna(0)

    # Calculate 'lonn_+_forbruk'
    rettes2["lonn_+_forbruk"] = rettes2["gjeldende_lonn_kr"] + rettes2["bedr_forbruk"]

    # Perform the if operation to update 'drkost_temp' based on the condition
    condition = rettes2["drkost_temp"] < rettes2["lonn_+_forbruk"]
    rettes2["drkost_temp"] = np.where(
        condition, rettes2["lonn_+_forbruk"], rettes2["drkost_temp"]
    )
    rettes2["theif"] = np.where(condition, 1, 0)

    
    # Filter 'rettes2' for rows where 'theif' is True within each 'orgnr_n_1' group
    dkvars = rettes2[rettes2.groupby("orgnr_n_1")["theif"].transform("any")]

    # Calculate 'utskudd' as the absolute difference between 'new_drkost', 'gjeldende_lonn_kr', and 'bedr_forbruk'
    dkvars["utskudd"] = (
        dkvars["new_drkost"] - dkvars["gjeldende_lonn_kr"] - dkvars["bedr_forbruk"]
    )
    dkvars["utskudd"] = abs(dkvars["utskudd"])

    # Keep selected columns
    columns_to_keep = [
        "orgnr_n_1",
        "lopenr",
        "radnr",
        "utskudd",
        "new_drkost",
        "drkost_temp",
        "theif",
        "gjeldende_lonn_kr",
        "bedr_forbruk",
    ]
    dkvars = dkvars[columns_to_keep]

    # Calculate sum of 'utskudd' grouped by 'orgnr_n_1', 'lopenr', and 'theif'
    sum7b = dkvars.groupby(["orgnr_n_1", "lopenr", "theif"])["utskudd"].sum().reset_index()

    # Pivot the 'sum7b' DataFrame to get 'thief0' and 'thief1' columns
    sum7b_transposed = sum7b.pivot(
        index=["orgnr_n_1", "lopenr"], columns="theif", values="utskudd"
    ).reset_index()

    # Rename columns as per SAS code
    sum7b_transposed.rename(columns={0: "thief0", 1: "thief1"}, inplace=True)
    sum7b_transposed = sum7b_transposed[["orgnr_n_1", "lopenr", "thief0", "thief1"]]

    # Merge 'sum7b_transposed' with 'dkvars' on 'orgnr_n_1' and 'lopenr'
    dkvars_2 = pd.merge(dkvars, sum7b_transposed, on=["orgnr_n_1", "lopenr"], how="inner")

    # Apply conditional logic for 'andel1' and 'andel2'
    pd.set_option("display.float_format", "{:.2f}".format)
    dkvars_2["andel1"] = np.where(
        dkvars_2["theif"] == 0, dkvars_2["utskudd"] / dkvars_2["thief0"], np.nan
    )
    dkvars_2["andel2"] = np.where(
        dkvars_2["theif"] == 0, np.round(dkvars_2["andel1"] * dkvars_2["thief1"]), np.nan
    )

    # Update 'new_drkost' based on 'andel2'
    dkvars_2["new_drkost"] = np.where(
        dkvars_2["theif"] == 0,
        dkvars_2["drkost_temp"] - dkvars_2["andel2"],
        dkvars_2["drkost_temp"],
    )

    # Keep selected columns for the final 'dkvars_3'
    columns_to_keep = ["orgnr_n_1", "lopenr", "radnr", "new_drkost"]
    dkvars_3 = dkvars_2[columns_to_keep]

    # Merge 'dkvars_3' with 'rettes2' on 'orgnr_n_1', 'lopenr', and 'radnr'
    merged_df = pd.merge(rettes2, dkvars_3, how='left', left_on=['orgnr_n_1', 'lopenr', 'radnr'], right_on=['orgnr_n_1', 'lopenr', 'radnr'], suffixes=('', '_updated'))

    # Delete unnecessary DataFrames to free up memory
    del rettes2, dkvars_3, dkvars_2

    # Update 'new_drkost' with values from 'new_drkost_updated' where available
    merged_df['new_drkost'] = merged_df['new_drkost_updated'].combine_first(merged_df['new_drkost'])

    # Identify duplicate IDs
    duplicate_ids = merged_df['id'][merged_df['id'].duplicated(keep=False)]

    # Update 'regtype' to '02' where 'regtype' is '01' and the ID appears more than once
    merged_df.loc[(merged_df['regtype'] == '01') & (merged_df['id'].isin(duplicate_ids)), 'regtype'] = '02'

    # Update specific columns where 'regtype' is '01'
    merged_df.loc[merged_df['regtype'] == '01', 'new_oms'] = merged_df['foretak_omsetning']
    merged_df.loc[merged_df['regtype'] == '01', 'new_drkost'] = merged_df['foretak_driftskostnad']
    merged_df.loc[merged_df['regtype'] == '01', 'bedr_salgsint'] = merged_df['salgsint']
    merged_df.loc[merged_df['regtype'] == '01', 'bedr_forbruk'] = merged_df['forbruk']

    # Group by 'id' and calculate the sum of 'new_drkost'
    test_grouped = (
        merged_df.groupby("id")[["new_drkost"]].sum().reset_index()
    )

    # Rename the columns
    test_grouped.rename(
        columns={"new_drkost": "test_tot_drkost_fordelt"},
        inplace=True,
    )

    # Merge 'test_grouped' with 'merged_df' on 'id'
    temp = pd.merge(merged_df, test_grouped, on="id", how="left")

    # Calculate the difference between 'foretak_driftskostnad' and 'test_tot_drkost_fordelt'
    temp['drkost_diff'] = temp['foretak_driftskostnad'] - temp['test_tot_drkost_fordelt']

    # Sort 'temp' by 'drkost_diff' in ascending order
    temp = temp.sort_values(by='drkost_diff', ascending=True)

    # Create a mask to filter rows where the absolute value of 'drkost_diff' is <= 1000
    mask = temp['drkost_diff'].abs() <= 1000

    # Create a new DataFrame with the rows to be checked manually
    check_manually = merged_df[~mask]

    # Update 'merged_df' to keep only the rows where the absolute value of 'drkost_diff' is <= 1000
    merged_df = merged_df[mask]

    # Create a copy of 'merged_df' to 'oppdateringsfil'
    oppdateringsfil = merged_df.copy()

    # Extract 'n3' and 'n2' from 'tmp_sn2007_5' in 'time_series_df'
    time_series_df["n3"] = time_series_df["tmp_sn2007_5"].str[:4]
    time_series_df["n2"] = time_series_df["tmp_sn2007_5"].str[:2]

    # Extract 'n2' from 'tmp_sn2007_5' in 'merged_df'
    merged_df["n2"] = merged_df["tmp_sn2007_5"].str[:2]

    # Select specific columns for 'temp_1' from 'time_series_df'
    temp_1 = time_series_df[['id',
                             'nacef_5',
                             'orgnr_n_1',
                             'b_sysselsetting_syss',
                             'b_kommunenr',
                             'gjeldende_lonn_kr', 
                             'gjeldende_driftsk_kr',
                             'gjeldende_omsetn_kr',
                             'tmp_forbruk_bed',
                             'tmp_salgsint_bed',
                             'tmp_sn2007_5',
                             'n3',
                             'n2',
                             'year']]

    # Rename columns in 'temp_1'
    temp_1 = temp_1.rename(columns={'b_sysselsetting_syss':'syss',
                                    'b_kommunenr':'kommunenr',
                                    'gjeldende_lonn_kr':'lonn',
                                    'gjeldende_omsetn_kr':'oms',
                                    'gjeldende_driftsk_kr': 'drkost',
                                    'tmp_forbruk_bed':'forbruk',
                                    'tmp_salgsint_bed':'salgsint',
                                   })

    # Filter out the current year from 'temp_1'
    temp_1 = temp_1[temp_1['year'] != year]

    # Extract 'n3' and 'n2' from 'tmp_sn2007_5' in 'merged_df'
    merged_df["n3"] = merged_df["tmp_sn2007_5"].str[:4]
    merged_df["n2"] = merged_df["tmp_sn2007_5"].str[:2]

    # Select specific columns for 'temp_2' from 'merged_df'
    temp_2 = merged_df[['id',
                        'nacef_5',
                        'orgnr_n_1',
                        'b_sysselsetting_syss',
                        'b_kommunenr',
                        'gjeldende_lonn_kr', 
                        'new_drkost',
                        'oms',
                        'bedr_forbruk',
                        'bedr_salgsint',
                        'tmp_sn2007_5',
                        'n3',
                        'n2',
                        'year']]

    # Rename columns in 'temp_2'
    temp_2 = temp_2.rename(columns={'b_sysselsetting_syss':'syss',
                                    'b_kommunenr':'kommunenr',
                                    'gjeldende_lonn_kr':'lonn',
                                    'bedr_forbruk':'forbruk',
                                    'bedr_salgsint':'salgsint',
                                    'new_drkost': 'drkost'
                                   })

    # Filter 'temp_2' for the current year
    temp_2 = temp_2[temp_2['year'] == year]

    # Fill NaN values in 'forbruk' and 'salgsint' with 0 in 'temp_1'
    temp_1['forbruk'] = temp_1['forbruk'].fillna(0)
    temp_1['salgsint'] = temp_1['salgsint'].fillna(0)

    # Concatenate 'temp_1' and 'temp_2' into 'timeseries_knn'
    timeseries_knn = pd.concat([temp_1, temp_2], axis=0)

    # Aggregate forbruk per year

    columns_to_convert = ['salgsint', 'forbruk', 'oms', 'drkost', 'lonn', 'syss']

    # Convert columns to numeric using pd.to_numeric for safe conversion, errors='coerce' will set issues to NaN
    for column in columns_to_convert:
        timeseries_knn[column] = pd.to_numeric(timeseries_knn[column], errors='coerce')

    # Convert 'year' and 'n3' to string
    timeseries_knn['year'] = timeseries_knn['year'].astype(str)
    timeseries_knn['n3'] = timeseries_knn['n3'].astype(str)

    # Calculate 'resultat' as the difference between 'oms' and 'drkost'
    timeseries_knn['resultat'] = timeseries_knn['oms'] - timeseries_knn['drkost']

    # Filter for n3 values in 45, 46, or 47
    timeseries_knn = timeseries_knn[timeseries_knn['n2'].isin(['45', '46', '47'])]
    temp = timeseries_knn.copy()

    # Aggregate data by 'year' and 'n3'
    timeseries_knn_agg = timeseries_knn.groupby(["year", "n3"])[["forbruk", "oms", "drkost", "salgsint", "lonn", 'syss', "resultat"]].sum().reset_index()

    # Calculate 'lonn_pr_syss' and 'oms_pr_syss'
    timeseries_knn_agg['lonn_pr_syss'] = timeseries_knn_agg['lonn'] / timeseries_knn_agg['syss']
    timeseries_knn_agg['oms_pr_syss'] = timeseries_knn_agg['oms'] / timeseries_knn_agg['syss']

    # Extract the first two characters of 'n3' to create 'n2'
    timeseries_knn_agg["n2"] = timeseries_knn_agg["n3"].str[:2]

    # Aggregate data by 'year', 'kommunenr', and 'n3'
    timeseries_knn__kommune_agg = temp.groupby(["year", "kommunenr", "n3"])[["forbruk", "oms", "drkost", "salgsint", "lonn", 'syss', "resultat"]].sum().reset_index()

    # Calculate 'lonn_pr_syss' and 'oms_pr_syss' for kommune-level data
    timeseries_knn__kommune_agg['lonn_pr_syss'] = timeseries_knn__kommune_agg['lonn'] / timeseries_knn__kommune_agg['syss']
    timeseries_knn__kommune_agg['oms_pr_syss'] = timeseries_knn__kommune_agg['oms'] / timeseries_knn__kommune_agg['syss']

    # Extract the first two characters of 'n3' to create 'n2' for kommune-level data
    timeseries_knn__kommune_agg["n2"] = timeseries_knn__kommune_agg["n3"].str[:2]

    # Create new columns 'n2_f' and 'n3_f' in 'oppdateringsfil'
    oppdateringsfil['n2_f'] = oppdateringsfil['nacef_5'].str[:2]
    oppdateringsfil['n3_f'] = oppdateringsfil['nacef_5'].str[:4]

    # Filter 'oppdateringsfil' for rows where 'n2_f' is in 45, 46, or 47
    oppdateringsfil = oppdateringsfil[oppdateringsfil['n2_f'].isin(['45', '46', '47'])]

    # Filter 'oppdateringsfil' for rows where 'radnr' is 1 to create 'foretak'
    temp = oppdateringsfil[oppdateringsfil['radnr'] == 1]

    # Group by 'n3_f' and sum specific columns in 'temp'
    temp = temp.groupby('n3_f').sum()[['foretak_omsetning', 'foretak_driftskostnad', 'forbruk', 'salgsint']].reset_index()

    # Group by 'n3_f' and sum specific columns in 'oppdateringsfil' to create 'bedrift'
    bedrift = oppdateringsfil.groupby('n3_f').sum()[['oms', 'new_drkost', 'bedr_forbruk', 'bedr_salgsint']].reset_index()

    # Merge 'temp' and 'bedrift' on 'n3_f' to create 'check_totals'
    check_totals = temp.merge(bedrift, on='n3_f', how='left')

    # Calculate processing time
    processing_time = time.time() - start_time
    print(f"Time taken to create training data: {processing_time:.2f} seconds")

    # Return the processed dataframes
    return oppdateringsfil, timeseries_knn_agg, timeseries_knn__kommune_agg, check_totals, check_manually

