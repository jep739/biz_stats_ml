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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
import geopandas as gpd
import sgis as sg
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from pyjstat import pyjstat
import sys
sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
fs = FileClient.get_gcs_file_system()
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing
import time
import kommune_translate


def main(year, limit):
    start_year = 2017
    all_good_dataframes = []  # List to store good dataframes for each year
    all_bad_dataframes = []   # List to store bad dataframes for each year
    all_training_dataframes = []  # List to store training dataframes for each year
    all_time_series_dataframes = []  # List to store time series dataframes for each year

    for current_year in range(start_year, year + 1):
        fjor = current_year - 1  # Previous year
        
        skjema_list = 'RA-0174-1'
        fil_path = [
            f
            for f in fs.glob(
                f"gs://ssb-prod-noeku-data-produkt/eimerdb/nokubasen/skjemadata/aar={current_year}/skjema={skjema_list}/*"
            )
            if f.endswith(".parquet")
        ]

        # Use the ParquetDataset to read multiple files
        dataset = pq.ParquetDataset(fil_path, filesystem=fs)
        table = dataset.read()

        # Convert to Pandas DataFrame
        skjema = table.to_pandas()
               
        felt_id_values = [
            "V_ORGNR",
            "F_ADRESSE",
            "FJOR_NACE_B_T1",
            "TMP_SN2007_5",
            "B_KOMMUNENR",
            "REGTYPE",
            "B_SYSSELSETTING_SYSS",
            "TMP_NY_BDR_SYSS",
            "GJELDENDE_BDR_SYSS",
            "FJOR_SYSSEL_T1",
            "LONN_PST_AORDN",
            "GJELDENDE_LONN_KR",
            "LONN",
            "FJOR_LONN_KR_T1",
            "TMP_SNITTLONN",
            "FJOR_SNITTLONN_T1",
            "GJELDENDE_OMSETN_KR",
            "OMSETN_KR",
            "FJOR_OMSETN_KR_T1",
            "TMP_SNITTOMS",
            "FJOR_SNITTOMS_T1",
            "TMP_SALGSINT_BED",
            "TMP_FORBRUK_BED",
            "VAREKOST_BED",
            "GJELDENDE_DRIFTSK_KR",
            "DRIFTSKOST_KR",
            "FJOR_DRIFTSKOST_KR_T1",
            "NACEF_5",
            "SALGSINT",
            "FORBRUK",
            "TMP_NO_P4005",
            "TMP_AVPROS_ORGFORB",
            "ORGNR_N_1",
            "TMP_NO_OMSETN",
            "TMP_DRIFTSKOSTNAD_9010",
            "TMP_DRIFTSKOSTNAD_9910",
        ]

        # Filter the DataFrame for the specified field values
        skjema = skjema[skjema["feltnavn"].isin(felt_id_values)]
        
        # Pivot the DataFrame
        skjema = skjema.pivot_table(
            index=["id", "radnr", "lopenr"],
            columns="feltnavn",
            values="feltverdi",
            aggfunc="first",
        )
        skjema = skjema.reset_index()
        skjema.columns = skjema.columns.str.lower()  # Convert column names to lower case
        
        # Foretak level data is always when radnr = 0
        foretak = skjema.loc[skjema["radnr"] == 0]

        # Create the 'bedrift' DataFrame
        bedrift = skjema.loc[skjema["radnr"] > 0]

        selected_columns = [
            "id",
            "lopenr",
            "forbruk",
            "nacef_5",
            "orgnr_n_1",
            "salgsint",
            "tmp_driftskostnad_9010",
            "tmp_driftskostnad_9910",
            "tmp_no_omsetn",
            "tmp_no_p4005",
        ]

        foretak = foretak[selected_columns]

        # Assuming 'foretak' is your DataFrame
        foretak.rename(columns={"tmp_no_omsetn": "foretak_omsetning"}, inplace=True)


        foretak = foretak.fillna(0)

        foretak[["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"]] = foretak[
            ["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"]
        ].apply(pd.to_numeric, errors="coerce")

        foretak["foretak_driftskostnad"] = foretak[
            ["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"]
        ].max(axis=1)

        # Drop the specified columns
        foretak.drop(["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"], axis=1, inplace=True)

        columns_to_drop = [
            "forbruk",
            "nacef_5",
            "orgnr_n_1",
            "salgsint",
            "tmp_driftskostnad_9010",
            "tmp_driftskostnad_9910",
            "tmp_no_omsetn",
            "tmp_no_p4005",
        ]

        bedrift.drop(columns_to_drop, axis=1, inplace=True)

        columns_to_fill = ["omsetn_kr", "driftskost_kr"]

        # Convert columns to numeric, replacing non-convertible values with NaN
        bedrift[columns_to_fill] = bedrift[columns_to_fill].apply(
            pd.to_numeric, errors="coerce"
        )

        # Fill NaN values with 0 for the specified columns
        bedrift[columns_to_fill] = bedrift[columns_to_fill].fillna(0)


        # hjelpe virksomheter
        if_condition = bedrift["regtype"] == "04"

        # If the condition is True, set 'omsetn_kr' equal to 'driftskost_kr'
        bedrift.loc[if_condition, "omsetn_kr"] = bedrift.loc[if_condition, "driftskost_kr"]


        # Group by 'id' and calculate the sum
        grouped_bedrift = (
            bedrift.groupby("id")[["omsetn_kr", "driftskost_kr"]].sum().reset_index()
        )

        # Rename the columns
        grouped_bedrift.rename(
            columns={"omsetn_kr": "tot_oms_fordelt", "driftskost_kr": "tot_driftskost_fordelt"},
            inplace=True,
        )

        # Merge the grouped DataFrame back to the original DataFrame based on 'id'
        bedrift = pd.merge(bedrift, grouped_bedrift, on="id", how="left")

        merged_df = pd.merge(foretak, bedrift, on=["id", "lopenr"], how="inner")
        
        
        # Convert columns to numeric, replacing non-convertible values with NaN
        merged_df["tot_oms_fordelt"] = pd.to_numeric(
            merged_df["tot_oms_fordelt"], errors="coerce"
        )
        merged_df["foretak_omsetning"] = pd.to_numeric(
            merged_df["foretak_omsetning"], errors="coerce"
        )

        # Calculate omsetning_percentage
        merged_df["omsetning_percentage"] = (
            merged_df["tot_oms_fordelt"] / merged_df["foretak_omsetning"]
        )

        # Convert columns to numeric, replacing non-convertible values with NaN
        merged_df["tot_driftskost_fordelt"] = pd.to_numeric(
            merged_df["tot_driftskost_fordelt"], errors="coerce"
        )
        merged_df["foretak_driftskostnad"] = pd.to_numeric(
            merged_df["foretak_driftskostnad"], errors="coerce"
        )

        # Calculate driftskostnader_percentage
        merged_df["driftskostnader_percentage"] = (
            merged_df["tot_driftskost_fordelt"] / merged_df["foretak_driftskostnad"]
        )

        merged_df["driftskostnader_percentage"] = (
            merged_df["tot_driftskost_fordelt"] / merged_df["foretak_driftskostnad"]
        ).round(4)

        # Fill NaN with a specific value (e.g., 0)
        merged_df["driftskostnader_percentage"].fillna(0, inplace=True)
        merged_df["omsetning_percentage"].fillna(0, inplace=True)
        
        # Create the 'Good' DataFrame
        good_temp_df = merged_df[
            (merged_df["omsetning_percentage"] >= limit)
            & (merged_df["driftskostnader_percentage"] >= limit)
        ]

        # Create 'bedrift_count' and 'distribution_count'
        good_temp_df["bedrift_count"] = good_temp_df.groupby("orgnr_n_1")[
            "orgnr_n_1"
        ].transform("count")
        good_temp_df["distribution_count"] = good_temp_df.groupby("orgnr_n_1")[
            "omsetn_kr"
        ].transform(lambda x: (x > 0).sum())

        # Create 'bad_temp' DataFrame based on conditions
        bad_temp = good_temp_df[
            (good_temp_df["bedrift_count"] > 5) & (good_temp_df["distribution_count"] <= 2)
        ]

        # Create 'good_df' by excluding rows from 'bad_temp'
        good_df = (
            pd.merge(good_temp_df, bad_temp, how="outer", indicator=True)
            .query('_merge == "left_only"')
            .drop("_merge", axis=1)
        )

        # Create the 'Mixed' DataFrame
        onlygoodoms = merged_df[
            (
                (merged_df["omsetning_percentage"] > limit)
                & (merged_df["driftskostnader_percentage"] <= limit)
            )
        ]

        onlygooddriftskostnader = merged_df[
            (
                (merged_df["driftskostnader_percentage"] > limit)
                & (merged_df["omsetning_percentage"] <= limit)
            )
        ]

        # Create the 'Bad' DataFrame
        bad_df = merged_df[
            (merged_df["omsetning_percentage"] <= limit)
            & (merged_df["driftskostnader_percentage"] <= limit)
        ]
        bad_df = pd.concat([bad_df, bad_temp]).drop_duplicates(keep=False)
        bad_df = pd.concat([bad_df, onlygooddriftskostnader]).drop_duplicates(keep=False)
        
        good_df = pd.concat([good_df, onlygoodoms]).drop_duplicates(keep=False)
        
        good_df["oms_share"] = good_df["omsetn_kr"] / good_df["tot_oms_fordelt"].round(5)

        # Round the values to whole numbers before assigning to the new columns
        good_df["new_oms"] = (
            (good_df["oms_share"] * good_df["foretak_omsetning"]).round(0).astype(int)
        )
        
        # swapped order of this concat , because we want even the bad data to remain in order to create trend lines. 
        bad_df["new_oms"] = bad_df["gjeldende_omsetn_kr"]
        merged_df = pd.concat([good_df, bad_df], ignore_index=True)
        
        time_series_df = merged_df.copy()

        # bad_df["new_oms"] = bad_df["gjeldende_omsetn_kr"]

        del onlygooddriftskostnader
        
        merged_df["n4"] = merged_df["nacef_5"].str[:5]


        kommune_befolk = kommune_pop.befolkning_behandling(current_year, fjor)
        kommune_inn = kommune_inntekt.inntekt_behandling(current_year, fjor)
        kpi_df = kpi.process_kpi_data(current_year)

        # Convert string columns to numeric
        merged_df["gjeldende_bdr_syss"] = pd.to_numeric(
            merged_df["gjeldende_bdr_syss"], errors="coerce"
        )
        merged_df["fjor_syssel_t1"] = pd.to_numeric(
            merged_df["fjor_syssel_t1"], errors="coerce"
        )

        # Perform division after conversion
        merged_df["emp_delta"] = merged_df["gjeldende_bdr_syss"] / merged_df["fjor_syssel_t1"]

        imputable_df = merged_df.copy()


        imputable_df = imputable_df.drop_duplicates(subset=["v_orgnr"])

        # imputable_df['n4'] =  imputable_df['nacef_5'].str[:5]
        imputable_df["n4"] = imputable_df["tmp_sn2007_5"].str[:5]

        imputable_df = pd.merge(imputable_df, kommune_befolk, on=["b_kommunenr"], how="left")
        imputable_df = pd.merge(imputable_df, kommune_inn, on=["b_kommunenr"], how="left")
        imputable_df = pd.merge(imputable_df, kpi_df, on=["n4"], how="left")

        # Ensure columns are numeric
        imputable_df["fjor_omsetn_kr_t1"] = pd.to_numeric(
            imputable_df["fjor_omsetn_kr_t1"], errors="coerce"
        )
        imputable_df["inflation_rate"] = pd.to_numeric(
            imputable_df["inflation_rate"], errors="coerce"
        )
        imputable_df["befolkning_delta"] = pd.to_numeric(
            imputable_df["befolkning_delta"], errors="coerce"
        )
        imputable_df["emp_delta"] = pd.to_numeric(imputable_df["emp_delta"], errors="coerce")
        imputable_df["inntekt_delta"] = pd.to_numeric(
            imputable_df["inntekt_delta"], errors="coerce"
        )

        general_inflation_rate = imputable_df.loc[
            imputable_df["n4"] == "47.78", "inflation_rate"
        ].values[0]
        imputable_df["inflation_rate"] = imputable_df["inflation_rate"].fillna(
            general_inflation_rate
        )

        imputable_df["inflation_rate_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["inflation_rate"]
        )
        imputable_df["befolkning_delta_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["befolkning_delta"]
        )
        imputable_df["emp_delta_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["emp_delta"]
        )
        imputable_df["inntekt_delta_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["inntekt_delta"]
        )

        # Treat Nan for inflation_rate_oms
        imputable_df["inflation_rate_oms"].replace([np.inf, -np.inf], np.nan, inplace=True)
        group_means = imputable_df.groupby("nacef_5")["inflation_rate_oms"].transform("mean")
        # Step 3: Fill NaN values in 'inflation_rate_oms' with the corresponding group's mean
        imputable_df["inflation_rate_oms"].fillna(group_means, inplace=True)


        categories_to_impute = [
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inntekt_delta_oms",
            "inflation_rate_oms",
        ]

        # Identify rows where 'b_sysselsetting_syss' is equal to 0
        rows_to_impute = imputable_df["b_sysselsetting_syss"] == 0

        # Replace NaN values with 0 for the identified rows and specified categories
        imputable_df.loc[rows_to_impute, categories_to_impute] = imputable_df.loc[
            rows_to_impute, categories_to_impute
        ].fillna(0)


        # Group by 'tmp_sn2007_5' and calculate the average 'emp_delta_oms'
        average_foretak_oms_pr_naring = imputable_df.groupby("tmp_sn2007_5")[
            "foretak_omsetning"
        ].mean()

        # Create a new column 'average_emp_delt_oms_pr_naring' and assign the calculated averages to it
        imputable_df["average_emp_delt_oms_pr_naring"] = imputable_df["nacef_5"].map(
            average_foretak_oms_pr_naring
        )

        # Fill NaN values with 0 before rounding and converting to int
        imputable_df["average_emp_delt_oms_pr_naring"] = imputable_df["average_emp_delt_oms_pr_naring"].fillna(0).round(0).astype(int)


        imputable_df["average_emp_delt_oms_pr_naring"] = (
            imputable_df["average_emp_delt_oms_pr_naring"].round(0).astype(int)
        )

        knn_df = imputable_df[
            [
                "average_emp_delt_oms_pr_naring",
                "emp_delta_oms",
                "befolkning_delta_oms",
                "inflation_rate_oms",
                "inntekt_delta_oms",
                "b_sysselsetting_syss",
                "v_orgnr",
            ]
        ]
        knn_df = knn_df.replace([np.inf, -np.inf], np.nan)


        imputable_df_copy = knn_df.copy()

        # Define the columns for numerical features
        numerical_features = [
            "average_emp_delt_oms_pr_naring",
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "inntekt_delta_oms",
            "b_sysselsetting_syss",
        ]

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), numerical_features)],
            remainder="passthrough",  # Keep non-numerical columns unchanged
        )

        knn_df["v_orgnr"] = knn_df["v_orgnr"].astype(str)

        # Ensure all columns in knn_df are numeric except 'v_orgnr'
        knn_df[numerical_features] = knn_df[numerical_features].apply(pd.to_numeric, errors="coerce")

        # Create KNN imputer
        knn_imputer = KNNImputer(n_neighbors=3)

        # Create imputer pipeline
        imputer_pipeline = Pipeline([("preprocessor", preprocessor), ("imputer", knn_imputer)])

        # Fit and transform the copy of your DataFrame
        imputed_values = imputer_pipeline.fit_transform(knn_df[numerical_features])

        # Convert the imputed values back to a DataFrame and merge 'v_orgnr' column
        imputed_knn_df = pd.DataFrame(imputed_values, columns=numerical_features)
        imputed_knn_df["v_orgnr"] = knn_df["v_orgnr"].values

        # Inverse transform the scaled numerical features
        inverse_scaled_features = preprocessor.named_transformers_["num"].inverse_transform(
            imputed_knn_df[numerical_features].values
        )
        imputed_knn_df[numerical_features] = inverse_scaled_features

        knn_df = imputed_knn_df.copy()

        # knn_df["v_orgnr"] = knn_df["v_orgnr"].round(0).astype(int)
        knn_df["v_orgnr"] = knn_df["v_orgnr"].astype(object)
        columns_to_drop = ["average_emp_delt_oms_pr_naring", "b_sysselsetting_syss"]
        knn_df.drop(columns=columns_to_drop, axis=1, inplace=True)


        columns_to_drop = [
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "inntekt_delta_oms",
        ]

        imputable_df.drop(columns=columns_to_drop, axis=1, inplace=True)


        knn_df["v_orgnr"] = knn_df["v_orgnr"].astype(str)
        imputable_df["v_orgnr"] = imputable_df["v_orgnr"].astype(str)

        # Strip 'v_orgnr' column in both knn_df and imputable_df
        knn_df["v_orgnr"] = knn_df["v_orgnr"].str.strip()
        imputable_df["v_orgnr"] = imputable_df["v_orgnr"].str.strip()

        imputable_df = pd.merge(imputable_df, knn_df, how="inner", on="v_orgnr")

        imputable_df_filtered = imputable_df[~imputable_df["regtype"].isin(["04", "11"])]


        columns_for_imputation = [
            "new_oms",
            "nacef_5",
            "inntekt_delta_oms",
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "v_orgnr",
            "gjeldende_bdr_syss",
        ]

        filtered_imputation_df = imputable_df_filtered[columns_for_imputation]
        filtered_imputation_df.replace([np.inf, -np.inf], np.nan, inplace=True)


        columns_to_drop = [
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "inntekt_delta_oms",
        ]

        imputable_df.drop(columns=columns_to_drop, axis=1, inplace=True)

        columns_for_imputation = filtered_imputation_df.columns.tolist()
        columns_for_imputation.remove("new_oms")

        # Filter for rows where all columns (except 'new_oms') have no NaN values
        cleaned_imputation_df = filtered_imputation_df.dropna(
            subset=columns_for_imputation, how="any"
        )

        # Filter for rows where at least one column (excluding 'new_oms') has NaN values
        nn_df = filtered_imputation_df[
            filtered_imputation_df[columns_for_imputation].isna().any(axis=1)
        ]

        cleaned_imputation_df["inflation_rate_oms"] = (
            cleaned_imputation_df["inflation_rate_oms"].round(0).astype(int)
        )
        cleaned_imputation_df["befolkning_delta_oms"] = (
            cleaned_imputation_df["befolkning_delta_oms"].round(0).astype(int)
        )
        cleaned_imputation_df["emp_delta_oms"] = (
            cleaned_imputation_df["emp_delta_oms"].round(0).astype(int)
        )
        cleaned_imputation_df["inntekt_delta_oms"] = (
            cleaned_imputation_df["inntekt_delta_oms"].round(0).astype(int)
        )

        filtered_indices = filtered_imputation_df.index.tolist()

        # Filter rows from filtered_imputation_df that are not present in cleaned_imputation_df (based on index)
        nn_df = filtered_imputation_df[
            ~filtered_imputation_df.index.isin(cleaned_imputation_df.index)
        ]

        training_data = pd.merge(
            cleaned_imputation_df,
            imputable_df[["v_orgnr", "tmp_sn2007_5", "b_kommunenr", "b_sysselsetting_syss", 'id', 'orgnr_n_1']],
            how="left",
            on=["v_orgnr"]
        )

        training_data["b_sysselsetting_syss"] = training_data["b_sysselsetting_syss"].fillna(0)

        training_data["b_sysselsetting_syss"] = pd.to_numeric(
            training_data["b_sysselsetting_syss"], errors="coerce"
        )

        # Now you can round the values and convert them to integers
        training_data["b_sysselsetting_syss"] = (
            training_data["b_sysselsetting_syss"].round(0).astype("Int64"))
        
        training_data['year'] = current_year
        good_df['year'] = current_year
        bad_df['year'] = current_year
        time_series_df['year'] = current_year
        
       
        # Create the DataFrames
        
        all_good_dataframes.append(good_df)
        all_bad_dataframes.append(bad_df)
        all_training_dataframes.append(training_data)
        all_time_series_dataframes.append(time_series_df)
        
    # Concatenate all DataFrames into a single DataFrame
    training_data = pd.concat(all_training_dataframes, ignore_index=True)
    bad_data = pd.concat(all_bad_dataframes, ignore_index=True)
    good_data = pd.concat(all_good_dataframes, ignore_index=True)
    time_series_df = pd.concat(all_time_series_dataframes, ignore_index=True)

    current_year_good_oms = good_data[good_data['year'] == year]
    current_year_bad_oms = bad_data[bad_data['year'] == year]
    v_orgnr_list_for_imputering = current_year_bad_oms['v_orgnr'].tolist()
    
    # Create trend data
    
    # Determine the number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")

    numerical_columns = [
        "new_oms"
    ]

    # Sort the data
    training_data = training_data.sort_values(by=["v_orgnr", "year"])

    # Function to process each group
    def process_group(v_orgnr, group):
        group_forecast = group[["v_orgnr", "year"]].copy()
        for col in numerical_columns:
            if col == "new_oms" and group[col].isna().any():
                group_forecast[f"{col}_trendForecast"] = np.nan
                continue
            X = group[group["year"] < year][["year"]]
            y = group[group["year"] < year][col]
            if len(X) > 1:
                model = LinearRegression()
                model.fit(X, y)
                current_year = pd.DataFrame({"year": [year]})
                forecast = model.predict(current_year)[0]
                group_forecast[f"{col}_trendForecast"] = model.predict(group[["year"]])
                group_forecast.loc[group_forecast["year"] == year, f"{col}_trendForecast"] = forecast
            else:
                group_forecast[f"{col}_trendForecast"] = np.nan
        return group_forecast

    # Parallel processing
    results = Parallel(n_jobs=num_cores)(
        delayed(process_group)(v_orgnr, group)
        for v_orgnr, group in training_data.groupby("v_orgnr")
    )

    # Concatenate results
    trend_forecasts = pd.concat(results, ignore_index=True)

    # Merge the trend forecasts with the original training data
    training_data = pd.merge(training_data, trend_forecasts, on=["v_orgnr", "year"], how="left")
    
    # Ensure 'new_oms' and 'gjeldende_bdr_syss' are numeric
    training_data['new_oms'] = pd.to_numeric(training_data['new_oms'], errors='coerce')
    training_data['gjeldende_bdr_syss'] = pd.to_numeric(training_data['gjeldende_bdr_syss'], errors='coerce')
    
    training_data = training_data[~training_data['v_orgnr'].isin(['111111111', '123456789'])]
    
    training_data = kommune_translate.translate_kommune_kodes_2(training_data)
    
    avg_new_oms_per_tmp_sn2007_5 = training_data.groupby(['tmp_sn2007_5', 'year']).apply(
        lambda x: (x['new_oms'] / x['gjeldende_bdr_syss']).replace([np.inf, -np.inf], np.nan).mean()).reset_index()
    avg_new_oms_per_tmp_sn2007_5.columns = ['tmp_sn2007_5', 'year', 'avg_new_oms_per_gjeldende_bdr_syss']


    # Calculate the average new_oms per gjeldende_bdr_syss for each tmp_sn2007_5, b_kommunenr, and year
    avg_new_oms_per_tmp_sn2007_5_per_b_kommunenr = training_data.groupby(['tmp_sn2007_5', 'b_kommunenr', 'year']).apply(
        lambda x: (x['new_oms'] / x['gjeldende_bdr_syss']).replace([np.inf, -np.inf], np.nan).mean()).reset_index()
    avg_new_oms_per_tmp_sn2007_5_per_b_kommunenr.columns = ['tmp_sn2007_5', 'b_kommunenr', 'year', 'avg_new_oms_per_gjeldende_bdr_syss_kommunenr']
    
    training_data = pd.merge(training_data, avg_new_oms_per_tmp_sn2007_5, on=['tmp_sn2007_5', 'year'], how='left')
    training_data = pd.merge(training_data, avg_new_oms_per_tmp_sn2007_5_per_b_kommunenr, on=['tmp_sn2007_5', 'b_kommunenr', 'year'], how='left')
    
    training_data['oms_syssmean_basedOn_naring'] = training_data['avg_new_oms_per_gjeldende_bdr_syss'] * training_data['gjeldende_bdr_syss']
    training_data['oms_syssmean_basedOn_naring_kommune'] = training_data['avg_new_oms_per_gjeldende_bdr_syss_kommunenr'] * training_data['gjeldende_bdr_syss']
    
    
    # Add geographical data:
    
    current_date = datetime.datetime.now()

    # Format the year and month
    current_year = current_date.strftime("%Y")
    current_year_int = int(current_date.strftime("%Y"))
    current_month = current_date.strftime("%m")

    # Subtract one day from the first day of the current month to get the last day of the previous month
    last_day_of_previous_month = datetime.datetime(
        current_date.year, current_date.month, 1
    ) - datetime.timedelta(days=1)

    # Now we can get the month number of the previous month
    previous_month = last_day_of_previous_month.strftime("%m")

    VOFSTI = "ssb-vof-data-delt-stedfesting-prod/klargjorte-data/parquet"


    dataframes = []

    for year in range(2017, current_year_int + 1):

        file_path = f"{VOFSTI}/stedfesting-situasjonsuttak_p{year}-{previous_month}_v1.parquet"


        vof_df = dp.read_pandas(f"{file_path}")
        vof_gdf = gpd.GeoDataFrame(
            vof_df,
            geometry=gpd.points_from_xy(
                vof_df["y_koordinat"],
                vof_df["x_koordinat"],
            ),
            crs=25833,
        )

        vof_gdf = vof_gdf.rename(
            columns={
                "orgnrbed": "v_orgnr",
                "org_nr": "orgnr_foretak",
                "nace1_sn07": "naring",
            }
        )

        vof_gdf = vof_gdf[
            [
                "v_orgnr",
                "orgnr_foretak",
                "naring",
                "x_koordinat",
                "y_koordinat",
                "rute_100m",
                "rute_1000m",
                "geometry",
            ]
        ]

        dataframes.append(vof_gdf)

    combined_gdf = pd.concat(dataframes, ignore_index=True)

    # Drop duplicate rows in the combined DataFrame
    combined_gdf = combined_gdf.drop_duplicates()

    training_data = pd.merge(training_data, combined_gdf, on="v_orgnr", how="left")
    
    temp = training_data.copy()
    
    temp.drop_duplicates(subset=['v_orgnr', 'id', 'year'], keep='first', inplace=True)
    
    merging_df = current_year_bad_oms[['v_orgnr', 'id', 'year', 'lopenr']]

    imputatable_df = pd.merge(merging_df, temp, on=['v_orgnr', 'id', 'year'], how='left')
      
    training_data = training_data[~training_data['v_orgnr'].isin(v_orgnr_list_for_imputering)]

    return current_year_good_oms, current_year_bad_oms, v_orgnr_list_for_imputering, training_data, imputatable_df, time_series_df
