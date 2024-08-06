import datetime
import getpass
import os

import dapla as dp
import gcsfs
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sgis as sg
from dapla import FileClient
from dapla.auth import AuthClient
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

fs = FileClient.get_gcs_file_system()
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def knn(aar, enhet):
    # Get the current date
    current_date = datetime.datetime.now()

    # Format the year and month
    current_year = current_date.strftime("%Y")
    current_month = current_date.strftime("%m")

    # Subtract one day from the first day of the current month to get the last day of the previous month
    last_day_of_previous_month = datetime.datetime(
        current_date.year, current_date.month, 1
    ) - datetime.timedelta(days=1)

    # Now we can get the month number of the previous month
    previous_month = last_day_of_previous_month.strftime("%m")

    VOFSTI = "ssb-vof-data-delt-stedfesting-prod/klargjorte-data/parquet"
    file_path = f"{VOFSTI}/stedfesting-situasjonsuttak_p{current_year}-{previous_month}_v1.parquet"

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
            "orgnrbed": "orgnr_bedrift",
            "org_nr": "orgnr_foretak",
            "nace1_sn07": "naring",
        }
    )

    vof_gdf = vof_gdf[
        [
            "orgnr_bedrift",
            "orgnr_foretak",
            "naring",
            "x_koordinat",
            "y_koordinat",
            "rute_100m",
            "rute_1000m",
            "geometry",
        ]
    ]
    pd.set_option("display.max_columns", None)

    vof_gdf = vof_gdf.dropna(subset=["x_koordinat"])
    vof_gdf = vof_gdf.drop_duplicates(subset="orgnr_bedrift")
    vof_gdf = vof_gdf.drop("orgnr_foretak", axis=1)

    # Bedrift data

    fil_path = f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{aar}/statistikkfil_bedrifter_pub.parquet"

    bedrifter = pd.read_parquet(fil_path, filesystem=fs)

    # Create 'nace4' by slicing the first 5 characters of 'naring'
    bedrifter["naring4"] = bedrifter["naring"].str[:5]

    bedrifter["naring_f_4"] = bedrifter["naring_f"].str[:5]

    # Create 'nace3' by slicing the first 4 characters of 'naring'
    bedrifter["naring3"] = bedrifter["naring"].str[:4]

    bedrifter["naring_f_3"] = bedrifter["naring_f"].str[:4]

    enhets_id = bedrifter.loc[bedrifter["orgnr_foretak"] == enhet, "enhets_id"].values[
        0
    ]

    # Assuming 'bedrifter' is your DataFrame, 'enhet' is your variable with the stored value

    # Create the 'naring3' list for rows where 'orgnr_foretak' equals 'enhet'
    naring3_list = list(
        bedrifter.loc[bedrifter["orgnr_foretak"] == enhet, "naring3"].unique()
    )

    # Create the 'naring_f_3' list for rows where 'orgnr_f' equals 'enhet'

    naring_f_3_list = list(
        bedrifter.loc[bedrifter["orgnr_foretak"] == enhet, "naring_f_3"].unique()
    )
    # Now you have two lists: 'naring3_list' and 'naring_f_3_list'

    # Assuming 'naring3_list' and 'naring_f_3_list' have been defined as per your previous instructions

    # Filter the DataFrame
    filtered_bedrifter = bedrifter[
        bedrifter["naring3"].isin(naring3_list)
        | bedrifter["naring_f_3"].isin(naring_f_3_list)
    ]

    filtered_bedrifter = filtered_bedrifter[
        [
            "orgnr_bedrift",
            "orgnr_foretak",
            "omsetning",
            "kommune",
            "naring_f",
            "naring4",
            "naring_f_4",
            "naring3",
            "naring_f_3",
            "nopost_driftskostnader",
            "sysselsetting_syss",
            "navn",
            "reg_type",
        ]
    ]

    # Calculate the counts for each 'orgnr_foretak' and create a new column 'bedrift_count'
    filtered_bedrifter["bedrift_count"] = filtered_bedrifter.groupby("orgnr_foretak")[
        "orgnr_foretak"
    ].transform("count")

    fil_path = f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{aar}/statistikkfil_foretak_pub.parquet"

    foretak = pd.read_parquet(fil_path, filesystem=fs)

    foretak = foretak[
        ["orgnr_foretak", "omsetning", "sysselsetting_syss", "nopost_driftskostnader"]
    ]

    foretak = foretak.rename(
        columns={
            "omsetning": "omsetning_foretak",
            "sysselsetting_syss": "sysselsetting_foretak",
            "nopost_driftskostnader": "nopost_driftskostnader_foretak",
        }
    )

    # Need to do this in order to calculate other variables. Not the true syss count.
    foretak["sysselsetting_foretak"].fillna(1, inplace=True)
    foretak["sysselsetting_foretak"].replace(0, 1, inplace=True)  # Replace 0 with 1

    filtered_bedrifter = filtered_bedrifter.merge(
        foretak, on="orgnr_foretak", how="left"
    )
    filtered_bedrifter = filtered_bedrifter.drop_duplicates(
        subset="orgnr_bedrift", keep="first"
    )

    filtered_bedrifter["sysselsetting_foretak"].replace(0, np.nan, inplace=True)
    filtered_bedrifter["bedrift_count"].replace(0, np.nan, inplace=True)

    # Create new columns with the calculated values
    filtered_bedrifter["oms_per_syss_foretak"] = (
        filtered_bedrifter["omsetning_foretak"]
        / filtered_bedrifter["sysselsetting_foretak"]
    )
    filtered_bedrifter["oms_per_bedrift_foretak"] = (
        filtered_bedrifter["omsetning_foretak"] / filtered_bedrifter["bedrift_count"]
    )

    merged_df = filtered_bedrifter.merge(vof_gdf, on="orgnr_bedrift", how="left")
    merged_gdf = gpd.GeoDataFrame(merged_df, geometry="geometry")
    merged_gdf = merged_gdf.dropna(subset=["x_koordinat", "y_koordinat"])

    enhet_df = merged_gdf[merged_gdf["orgnr_foretak"] == enhet]

    features = [
        "x_koordinat",
        "y_koordinat",
        "sysselsetting_syss",
        "naring",
        "naring3",
        "omsetning_foretak",
        "nopost_driftskostnader_foretak",
        "nopost_driftskostnader_foretak",
        "oms_per_syss_foretak",
        "oms_per_bedrift_foretak",
    ]

    non_feature_columns = [
        "orgnr_foretak",
        "orgnr_bedrift",
    ]  # Non-feature columns to keep

    merged_gdf = merged_gdf.dropna(subset=features + non_feature_columns)

    from sklearn.compose import ColumnTransformer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Assume 'merged_gdf' is your DataFrame and it has been preprocessed to include the following columns:
    # 'x_koordinat', 'y_koordinat', 'sysselsetting_syss', 'naring', 'omsetning'
    # Preprocessing
    # StandardScaler will scale the coordinates and number of employees
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [
                    "x_koordinat",
                    "y_koordinat",
                    "sysselsetting_syss",
                    "omsetning_foretak",
                    "nopost_driftskostnader_foretak",
                    "oms_per_syss_foretak",
                    "oms_per_bedrift_foretak",
                ],
            ),
            ("cat", OneHotEncoder(), ["naring", "naring3"]),  # One-hot encode 'naring'
        ]
    )

    # Model setup
    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsRegressor(n_neighbors=5)),  # Adjust n_neighbors as needed
        ]
    )

    # Fit the model
    features = [
        "x_koordinat",
        "y_koordinat",
        "sysselsetting_syss",
        "naring",
        "naring3",
        "omsetning_foretak",
        "nopost_driftskostnader_foretak",
        "oms_per_syss_foretak",
        "oms_per_bedrift_foretak",
    ]
    target = "omsetning"

    knn_pipeline.fit(merged_gdf[features], merged_gdf[target])

    # To predict, you would follow a similar approach to the previous examples,
    # ensuring that you pass the x and y coordinates directly to the model after scaling.

    # Replace 'target_variable_name' with the actual name of your target variable
    features = [
        "x_koordinat",
        "y_koordinat",
        "sysselsetting_syss",
        "naring",
        "omsetning_foretak",
        "nopost_driftskostnader_foretak",
        "oms_per_syss_foretak",
        "oms_per_bedrift_foretak",
    ]
    target = "omsetning"

    # Preprocessing
    # Define the preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [
                    "x_koordinat",
                    "y_koordinat",
                    "sysselsetting_syss",
                    "omsetning_foretak",
                    "nopost_driftskostnader_foretak",
                    "oms_per_syss_foretak",
                    "oms_per_bedrift_foretak",
                ],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["naring"],
            ),
        ]
    )

    # Model setup
    # Setup the pipeline with the preprocessing and the KNN model
    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
        ]
    )

    # GroupShuffleSplit
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in group_split.split(
        merged_gdf, groups=merged_gdf["orgnr_foretak"]
    ):
        X_train, X_test = merged_gdf.iloc[train_idx], merged_gdf.iloc[test_idx]
        y_train, y_test = (
            merged_gdf[target].iloc[train_idx],
            merged_gdf[target].iloc[test_idx],
        )

    # Train the model
    knn_pipeline.fit(X_train[features], y_train)

    # Make predictions on the test set
    y_pred = knn_pipeline.predict(X_test[features])

    adjusted_result_df = pd.DataFrame(
        X_test, columns=features + ["orgnr_bedrift", "reg_type", "orgnr_foretak"]
    )
    adjusted_result_df["actual_omsetning"] = y_test

    # Add the predicted 'omsetning' from y_pred
    adjusted_result_df["predicted_omsetning"] = y_pred

    total_predicted = (
        adjusted_result_df.groupby("orgnr_foretak")["predicted_omsetning"]
        .sum()
        .reset_index()
    )
    total_predicted.rename(
        columns={"predicted_omsetning": "total_predicted_for_foretak"}, inplace=True
    )

    adjusted_result_df = adjusted_result_df.merge(
        total_predicted, on="orgnr_foretak", how="left"
    )

    # Calculate the percentage share of each 'orgnr_bedrift' prediction relative to its 'orgnr_foretak'
    adjusted_result_df["predicted_share"] = (
        adjusted_result_df["predicted_omsetning"]
        / adjusted_result_df["total_predicted_for_foretak"]
    )

    # Calculate the new adjusted predicted 'omsetning' based on the percentage share
    adjusted_result_df["new_predicted"] = (
        adjusted_result_df["predicted_share"] * adjusted_result_df["omsetning_foretak"]
    )

    # Now adjusted_result_df contains the adjusted predictions displayed as whole numbers
    adjusted_result_df.head(50)

    # Filter the DataFrame to include only rows where reg_type is '02'
    reg_type_02_df = adjusted_result_df[adjusted_result_df["reg_type"] == "02"]

    reg_type_02_df = reg_type_02_df.sort_values(by="orgnr_foretak")

    adjusted_result_df = adjusted_result_df.fillna(0)

    # Define your features as before
    features = [
        "x_koordinat",
        "y_koordinat",
        "oms_per_syss_foretak",
        "oms_per_bedrift_foretak",
    ]

    # units of a foretak cant be neighbors to each other:

    # Filter out the instances of the specific 'orgnr_foretak'
    orgnr_foretak_value = f"{enhet}"  # The 'orgnr_foretak' you're interested in
    filtered_gdf = merged_gdf[merged_gdf["orgnr_foretak"] != orgnr_foretak_value]

    # Fit the NearestNeighbors model on the filtered DataFrame
    nn = NearestNeighbors()
    nn.fit(filtered_gdf[features])

    # Find the instances of the specific 'orgnr_foretak' in the original DataFrame
    specific_orgnr_indices = merged_gdf.index[
        merged_gdf["orgnr_foretak"] == orgnr_foretak_value
    ].tolist()

    # For each instance of 'orgnr_foretak', find its nearest neighbors in the filtered DataFrame
    all_neighbors_indices = []
    for index in specific_orgnr_indices:
        distances, indices = nn.kneighbors(
            [merged_gdf.loc[index, features].values], n_neighbors=5
        )
        all_neighbors_indices.extend(indices.flatten())

    # Remove duplicates from the neighbors' indices
    all_neighbors_indices = list(set(all_neighbors_indices))

    # Create a DataFrame of the nearest neighbors
    neighbors_df = filtered_gdf.iloc[all_neighbors_indices]

    # Combine the specific 'orgnr_foretak' rows with the neighbors
    naermeste_naboer = pd.concat(
        [merged_gdf.iloc[specific_orgnr_indices], neighbors_df]
    )

    # Continue with your analysis using the filtered DataFrame
    return enhet_df, merged_gdf, reg_type_02_df, adjusted_result_df, naermeste_naboer


enhet_df, merged_gdf, reg_type_02_df, adjusted_result_df, naermeste_naboer = knn("2021", "879263662")

merged_gdf


