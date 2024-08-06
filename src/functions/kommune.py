def kommune(variable, naring, year, df):
    
    import pandas as pd
    import sgis as sg

    # Define the path to the test data
    testdatasti = "ssb-prod-dapla-felles-data-delt/GIS/testdata"

    # Read the geopandas dataframe for kommuner
    kommuner = sg.read_geopandas(f"{testdatasti}/enkle_kommuner.parquet")

    # Filter the dataframe for the specified year
    df = df[df['year'] == year]

    # Group by 'n3' and 'b_kommunenr', and then sum up the specified variable
    aggregated_df = df.groupby(['n3', 'kommunenr'])[variable].sum().reset_index()

    # Filter the aggregated dataframe where 'n3' equals the specified naring
    filtered_df = aggregated_df[aggregated_df['n3'] == naring]

    # Rename columns for merging
    filtered_df = filtered_df.rename(columns={'kommunenr': 'KOMMUNENR'})

    # Select only the necessary columns
    filtered_df = filtered_df[['KOMMUNENR', variable]]

    # Ensure 'KOMMUNENR' columns are string types and properly formatted
    kommuner["KOMMUNENR"] = kommuner["KOMMUNENR"].str.replace('"', "").astype(str)
    filtered_df["KOMMUNENR"] = filtered_df["KOMMUNENR"].astype(str).str.zfill(4)

    # Merge the kommuner dataframe with the filtered dataframe on 'KOMMUNENR'
    kommuner = pd.merge(kommuner, filtered_df, on="KOMMUNENR", how="left")

    # Fill NaN values in the specified variable with 0
    kommuner[variable] = kommuner[variable].fillna(0)

    return kommuner


def get_coordinates(df):
    
    import pandas as pd
    import geopandas as gpd
    import sgis as sg
    import dapla as dp
    import datetime
    from dapla.auth import AuthClient
    from dapla import FileClient

    # Add geographical data:
    
    # Get the current date
    current_date = datetime.datetime.now()

    # Format the current year and month
    current_year = current_date.strftime("%Y")
    current_year_int = int(current_date.strftime("%Y"))
    current_month = current_date.strftime("%m")

    # Subtract one day from the first day of the current month to get the last day of the previous month
    last_day_of_previous_month = datetime.datetime(
        current_date.year, current_date.month, 1
    ) - datetime.timedelta(days=1)

    # Get the month number of the previous month
    previous_month = last_day_of_previous_month.strftime("%m")

    # Define the path to the VOF data
    VOFSTI = "ssb-vof-data-delt-stedfesting-prod/klargjorte-data/parquet"

    # Initialize a list to store dataframes
    dataframes = []

    # Loop through the years from 2017 to the current year
    for year in range(2017, current_year_int + 1):
        # Define the file path for the current year and previous month
        file_path = f"{VOFSTI}/stedfesting-situasjonsuttak_p{year}-{previous_month}_v1.parquet"

        # Read the data into a pandas DataFrame
        vof_df = dp.read_pandas(f"{file_path}")

        # Convert the DataFrame to a GeoDataFrame
        vof_gdf = gpd.GeoDataFrame(
            vof_df,
            geometry=gpd.points_from_xy(
                vof_df["y_koordinat"],
                vof_df["x_koordinat"],
            ),
            crs=25833,
        )

        # Rename columns for consistency
        vof_gdf = vof_gdf.rename(
            columns={
                "orgnrbed": "v_orgnr",
                "org_nr": "orgnr_foretak",
                "nace1_sn07": "naring",
            }
        )

        # Select specific columns to keep
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

        # Append the GeoDataFrame to the list
        dataframes.append(vof_gdf)

    # Combine all the GeoDataFrames into one
    combined_gdf = pd.concat(dataframes, ignore_index=True)

    # Drop duplicate rows in the combined GeoDataFrame
    combined_gdf = combined_gdf.drop_duplicates()

    # Merge the combined GeoDataFrame with the input DataFrame on 'v_orgnr'
    df = pd.merge(df, combined_gdf, on="v_orgnr", how="left")

    # Convert the merged DataFrame to a GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(df, geometry="geometry")

    # Drop rows where 'x_koordinat' or 'y_koordinat' are NaN
    merged_gdf = merged_gdf.dropna(subset=["x_koordinat", "y_koordinat"])

    return merged_gdf
