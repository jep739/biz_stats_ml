import datetime
import getpass
import os
import sys

# import sgis as sg
import dapla as dp
import gcsfs
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dapla import FileClient
from dapla.auth import AuthClient
from pyjstat import pyjstat
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append("../functions")
import ao
import kommune_inntekt
import kommune_pop
import kpi

fs = FileClient.get_gcs_file_system()
import numpy as np


def split_input_data(year, limit):

    fjor = year - 1

    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/eimerdb/nokubasen/skjemadata/aar={year}/skjema=RA-0174-1/*"
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

    # Pivot the df
    skjema = skjema[skjema["feltnavn"].isin(felt_id_values)]

    pivot_df = skjema.pivot_table(
        index=["id", "radnr", "lopenr"],
        columns="feltnavn",
        values="feltverdi",
        aggfunc="first",
    )
    pivot_df = pivot_df.reset_index()
    pivot_df.columns = pivot_df.columns.str.lower()

    # Create new variable 'year'
    pivot_df["year"] = year

    # Seperate foretak and bedrifter
    foretak = pivot_df.loc[pivot_df["radnr"] == 0]

    # Create the 'bedrift' DataFrame
    bedrift = pivot_df.loc[pivot_df["radnr"] > 0]

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
    foretak.drop(
        ["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"], axis=1, inplace=True
    )

    # Assuming 'bedrift' is your DataFrame
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

    # Assuming 'bedrift' is your DataFrame
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
        columns={
            "omsetn_kr": "tot_oms_fordelt",
            "driftskost_kr": "tot_driftskost_fordelt",
        },
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

    good_df["oms_share"] = good_df["omsetn_kr"] / good_df["tot_oms_fordelt"].round(5)

    # Round the values to whole numbers before assigning to the new columns
    good_df["new_oms"] = (
        (good_df["oms_share"] * good_df["foretak_omsetning"]).round(0).astype(int)
    )

    good_df["oms_share"] = good_df["new_oms"] / good_df["tot_oms_fordelt"].round(5)

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

    del onlygooddriftskostnader

    return good_df, onlygoodoms, bad_df, merged_df


def create_training_data(year, limit):

    fjor = year - 1

    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/eimerdb/nokubasen/skjemadata/aar={year}/skjema=RA-0174-1/*"
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

    # Pivot the df
    skjema = skjema[skjema["feltnavn"].isin(felt_id_values)]

    pivot_df = skjema.pivot_table(
        index=["id", "radnr", "lopenr"],
        columns="feltnavn",
        values="feltverdi",
        aggfunc="first",
    )
    pivot_df = pivot_df.reset_index()
    pivot_df.columns = pivot_df.columns.str.lower()

    # Create new variable 'year'
    pivot_df["year"] = year

    # Seperate foretak and bedrifter
    foretak = pivot_df.loc[pivot_df["radnr"] == 0]

    # Create the 'bedrift' DataFrame
    bedrift = pivot_df.loc[pivot_df["radnr"] > 0]

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
    foretak.drop(
        ["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"], axis=1, inplace=True
    )

    # Assuming 'bedrift' is your DataFrame
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

    # Assuming 'bedrift' is your DataFrame
    columns_to_fill = ["omsetn_kr", "driftskost_kr"]

    # Convert columns to numeric, replacing non-convertible values with NaN
    bedrift[columns_to_fill] = bedrift[columns_to_fill].apply(
        pd.to_numeric, errors="coerce"
    )

    # Fill NaN values with 0 for the specified columns
    bedrift[columns_to_fill] = bedrift[columns_to_fill].fillna(0)

    #     # hjelpe virksomheter
    #     if_condition = bedrift['regtype'] == '04'

    #     # If the condition is True, set 'omsetn_kr' equal to 'driftskost_kr'
    #     bedrift.loc[if_condition, 'omsetn_kr'] = bedrift.loc[if_condition, 'driftskost_kr']

    # Group by 'id' and calculate the sum
    grouped_bedrift = (
        bedrift.groupby("id")[["omsetn_kr", "driftskost_kr"]]
        .sum()
        .reset_index()
    )

    # Rename the columns
    grouped_bedrift.rename(
        columns={
            "omsetn_kr": "tot_oms_fordelt",
            "driftskost_kr": "tot_driftskost_fordelt",
        },
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


    good_df["oms_share"] = good_df["omsetn_kr"] / good_df["tot_oms_fordelt"].round(5)

    # Round the values to whole numbers before assigning to the new columns
    good_df["new_oms"] = (
        (good_df["oms_share"] * good_df["foretak_omsetning"]).round(0).astype(int)
    )

    good_df["oms_share"] = good_df["new_oms"] / good_df["tot_oms_fordelt"].round(5)


    # Create the 'Mixed' DataFrame
    onlygoodoms = merged_df[
        (
            (merged_df["omsetning_percentage"] > limit)
            & (merged_df["driftskostnader_percentage"] <= limit)
        )
    ]
    
    training_data = pd.concat([good_df, onlygoodoms]).drop_duplicates(keep=False)
    
    # Assuming 'imputer' is your DataFrame and you want to select specific columns
    selected_columns = [
        "id",
        "lopenr",
        "forbruk",
        "nacef_5",
        "orgnr_n_1",
        "salgsint",
        "foretak_omsetning",
        "tmp_no_p4005",
        "foretak_driftskostnad",
        "radnr",
        "gjeldende_bdr_syss",
        "fjor_driftskost_kr_t1",
        "fjor_lonn_kr_t1",
        "fjor_syssel_t1",
        "gjeldende_lonn_kr",
        "new_oms",
        "b_kommunenr",
    ]

    # Create a new DataFrame with only the selected columns
    imputer = training_data[selected_columns].copy()

    training_data["n4"] = training_data["nacef_5"].str[:5]
    
    kommune_befolk = kommune_pop.befolkning_behandling(year, fjor)
    kommune_inn = kommune_inntekt.inntekt_behandling(year, fjor)
    kpi_df = kpi.process_kpi_data(year)
    
    # Convert string columns to numeric
    training_data["gjeldende_bdr_syss"] = pd.to_numeric(
        training_data["gjeldende_bdr_syss"], errors="coerce"
    )
    training_data["fjor_syssel_t1"] = pd.to_numeric(
        training_data["fjor_syssel_t1"], errors="coerce"
    )

    # Perform division after conversion
    training_data["emp_delta"] = training_data["gjeldende_bdr_syss"] / training_data["fjor_syssel_t1"]
    
    imputable_df = training_data.copy()


    imputable_df = imputable_df.drop_duplicates(subset=["v_orgnr"])

    # imputable_df['n4'] =  imputable_df['nacef_5'].str[:5]
    imputable_df["n4"] = imputable_df["tmp_sn2007_5"].str[:5]

    imputable_df = pd.merge(imputable_df, kommune_befolk, on="b_kommunenr", how="left")
    imputable_df = pd.merge(imputable_df, kommune_inn, on="b_kommunenr", how="left")
    imputable_df = pd.merge(imputable_df, kpi_df, on="n4", how="left")

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

    # imputable_df['inflation_rate_oms'] = imputable_df['inflation_rate_oms'].round(0).astype(int)
    # imputable_df['befolkning_delta_oms'] = imputable_df['befolkning_delta_oms'].round(0).astype(int)
    # imputable_df['emp_delta_oms'] = imputable_df['emp_delta_oms'].round(0).astype(int)
    # imputable_df['inntekt_delta_oms'] = imputable_df['inntekt_delta_oms'].round(0).astype(int)

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
    imputable_df["average_emp_delt_oms_pr_naring"] = (
        imputable_df["average_emp_delt_oms_pr_naring"].round(0).astype(int)
    )
    
    imputable_df = imputable_df[~imputable_df["regtype"].isin(["04", "11"])]
    
    imputable_df["inflation_rate_oms"] = (
        imputable_df["inflation_rate_oms"].round(0).astype(int)
    )
    imputable_df["befolkning_delta_oms"] = (
        imputable_df["befolkning_delta_oms"].round(0).astype(int)
    )
    imputable_df["emp_delta_oms"] = (
        imputable_df["emp_delta_oms"].round(0).astype(int)
    )
    imputable_df["inntekt_delta_oms"] = (
        imputable_df["inntekt_delta_oms"].round(0).astype(int)
)

    return imputable_df
