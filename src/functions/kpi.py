import pandas as pd
import pyjstat
import requests


def fetch_kpi_data(year):

    POST_URL = "https://data.ssb.no/api/v0/no/table/03013/"
    bef_kom = {
        "query": [
            {
                "code": "Konsumgrp",
                "selection": {"filter": "vs:CoiCop2016niva1", "values": []},
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["Tolvmanedersendring"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": [str(year) + "M12"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    # Behandler spørringen
    resultat1 = requests.post(POST_URL, json=bef_kom)

    # Check if request was successful
    if resultat1.status_code == 200:
        from pyjstat import pyjstat

        # Convert JSON response to DataFrame
        dataset1 = pyjstat.Dataset.read(resultat1.text)
        df_temp = dataset1.write("dataframe")

        # Extract value from the DataFrame
        value = df_temp.iloc[0]["value"]

        # Create a DataFrame with 'konsumgrp' and 'value' columns
        df = pd.DataFrame(
            {
                "konsumgrp": [
                    "47.78",
                    "47.79",
                    "47.89",
                    "47.91",
                    "47.92",
                    "45.40",
                    "46.11",
                    "46.12",
                    "46.13",
                    "46.14",
                    "46.15",
                    "46.16",
                    "46.17",
                    "46.18",
                    "46.19",
                    "46.21",
                    "46.23",
                    "46.24",
                    "46.48",
                    "46.61",
                    "46.62",
                    "46.63",
                    "46.64",
                    "46.65",
                    "46.66",
                    "46.69",
                    "46.72",
                    "46.73",
                    "46.74",
                    "46.75",
                    "46.76",
                    "46.77",
                    "46.90",
                ],
                "value": [value] * 33,  # Repeat the value for each konsumgrp
            }
        )

        return df
    else:
        print("Failed to retrieve data:", resultat1.status_code)
        return None


def fetch_hovedgruppe_data(year):

    POST_URL = "https://data.ssb.no/api/v0/no/table/03013/"
    bef_kom = {
        "query": [
            {
                "code": "Konsumgrp",
                "selection": {
                    "filter": "vs:CoiCop2016niva2",
                    "values": ["01", "02", "03", "05", "08"],
                },
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["Tolvmanedersendring"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": [str(year) + "M12"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    # Behandler spørringen
    resultat1 = requests.post(POST_URL, json=bef_kom)

    # Check if request was successful
    if resultat1.status_code == 200:
        from pyjstat import pyjstat

        # Convert JSON response to DataFrame
        dataset1 = pyjstat.Dataset.read(resultat1.text)
        df = dataset1.write("dataframe")

        # Add 'region' column based on the values in the query
        konsumgrp = bef_kom["query"][0]["selection"]["values"]
        df["konsumgrp"] = konsumgrp

        konsumgrp_mapping = {
            "03": ["47.82", "46.42"],
            "05": ["47.52", "46.49"],
            "08": ["47.42", "46.52"],
        }

        # Split rows for "07.1.1"
        new_rows = []
        for _, row in df.iterrows():
            if row["konsumgrp"] in konsumgrp_mapping:
                new_value = konsumgrp_mapping[row["konsumgrp"]]
                if isinstance(new_value, list):
                    for val in new_value:
                        new_row = row.copy()
                        new_row["konsumgrp"] = val
                        new_row["value"] = row["value"] / len(
                            new_value
                        )  # Distribute value equally
                        new_rows.append(new_row)
                else:
                    row["konsumgrp"] = new_value
                    new_rows.append(row)
            else:
                new_rows.append(row)

        df = pd.DataFrame(new_rows)

        # df['konsumgrp'] = df['konsumgrp'].replace(konsumgrp_mapping)

        df["value"] = pd.to_numeric(df["value"])

        # Calculate average of "01" and "02" values
        avg_01_02 = (
            df[df["konsumgrp"] == "01"]["value"].iloc[0]
            + df[df["konsumgrp"] == "02"]["value"].iloc[0]
        ) / 2

        # Create a new DataFrame entry for the averaged value
        new_entry1 = pd.DataFrame({"konsumgrp": ["47.11"], "value": [avg_01_02]})
        new_entry2 = pd.DataFrame({"konsumgrp": ["47.19"], "value": [avg_01_02]})
        new_entry3 = pd.DataFrame({"konsumgrp": ["47.81"], "value": [avg_01_02]})

        # Concatenate the new entry with the original DataFrame
        df = pd.concat([df, new_entry1, new_entry2, new_entry3], ignore_index=True)

        # Reorder columns
        df = df[["konsumgrp", "value"]]

        return df
    else:
        print("Failed to retrieve data:", resultat1.status_code)
        return None


def fetch_gruppe_data(year):

    POST_URL = "https://data.ssb.no/api/v0/no/table/03013/"
    bef_kom = {
        "query": [
            {
                "code": "Konsumgrp",
                "selection": {
                    "filter": "vs:CoiCop2016niva3",
                    "values": [
                        "01.2",
                        "02.1",
                        "02.2",
                        "03.1",
                        "03.2",
                        "04.5",
                        "05.1",
                        "06.1",
                        "09.1",
                        "09.3",
                        "12.1",
                        "12.3",
                    ],
                },
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["Tolvmanedersendring"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": [str(year) + "M12"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    # Behandler spørringen
    resultat1 = requests.post(POST_URL, json=bef_kom)

    # Check if request was successful
    if resultat1.status_code == 200:
        from pyjstat import pyjstat

        # Convert JSON response to DataFrame
        dataset1 = pyjstat.Dataset.read(resultat1.text)
        df = dataset1.write("dataframe")

        konsumgrp = bef_kom["query"][0]["selection"]["values"]
        df["konsumgrp"] = konsumgrp

        konsumgrp_mapping = {
            "02.2": ["47.26", "46.35"],
            "03.1": "47.71",
            "03.2": "47.72",
            "04.5": "46.71",
            "05.1": ["47.53", "46.47"],
            "06.1": "47.74",
            "09.1": "47.43",
            "09.3": "47.76",
            "12.1": ["47.75", "46.45"],
            "12.3": "47.77",
        }

        # Split rows for "07.1.1"
        new_rows = []
        for _, row in df.iterrows():
            if row["konsumgrp"] in konsumgrp_mapping:
                new_value = konsumgrp_mapping[row["konsumgrp"]]
                if isinstance(new_value, list):
                    for val in new_value:
                        new_row = row.copy()
                        new_row["konsumgrp"] = val
                        new_row["value"] = row["value"] / len(
                            new_value
                        )  # Distribute value equally
                        new_rows.append(new_row)
                else:
                    row["konsumgrp"] = new_value
                    new_rows.append(row)
            else:
                new_rows.append(row)

        df = pd.DataFrame(new_rows)

        # df['konsumgrp'] = df['konsumgrp'].replace(konsumgrp_mapping)

        df["value"] = pd.to_numeric(df["value"])

        # Calculate average of "01" and "02" values
        avg_01_02 = (
            df[df["konsumgrp"] == "01.2"]["value"].iloc[0]
            + df[df["konsumgrp"] == "02.1"]["value"].iloc[0]
        ) / 2

        # Create a new DataFrame entry for the averaged value
        new_entry1 = pd.DataFrame({"konsumgrp": ["47.25"], "value": [avg_01_02]})

        new_entry2 = pd.DataFrame({"konsumgrp": ["46.34"], "value": [avg_01_02]})

        # Concatenate the new entry with the original DataFrame
        df = pd.concat([df, new_entry1, new_entry2], ignore_index=True)

        # Reorder columns
        df = df[["konsumgrp", "value"]]

        return df
    else:
        print("Failed to retrieve data:", resultat1.status_code)
        return None


def fetch_subgruppe1_data(year):

    POST_URL = "https://data.ssb.no/api/v0/no/table/03013/"
    bef_kom = {
        "query": [
            {
                "code": "Konsumgrp",
                "selection": {
                    "filter": "vs:CoiCop2016niva4",
                    "values": [
                        "01.1.1",
                        "01.1.2",
                        "01.1.3",
                        "01.1.4",
                        "01.1.5",
                        "01.1.6",
                        "01.1.7",
                        "01.1.8",
                        "01.1.9",
                        "01.2.1",
                        "05.1.1",
                        "05.2.0",
                        "05.3.2",
                        "06.1.1",
                        "07.1.1",
                        "07.2.1",
                        "07.2.2",
                        "07.2.3",
                        "09.1.3",
                        "09.1.4",
                        "09.3.1",
                        "09.3.2",
                        "09.3.3",
                        "09.5.1",
                        "09.5.2",
                    ],
                },
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["Tolvmanedersendring"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": [str(year) + "M12"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    # Behandler spørringen
    resultat1 = requests.post(POST_URL, json=bef_kom)

    # Check if request was successful
    if resultat1.status_code == 200:
        from pyjstat import pyjstat

        # Convert JSON response to DataFrame
        dataset1 = pyjstat.Dataset.read(resultat1.text)
        df = dataset1.write("dataframe")

        konsumgrp_mapping = {
            "01.1.2": ["47.22", "46.32"],
            "01.1.3": ["47.23", "46.38"],
            "01.1.8": "46.36",
            "01.1.9": ["47.29", "46.39"],
            "01.2.1": "46.37",
            "05.1.1": "47.59",
            "05.2.0": ["47.51", "46.41"],
            "05.3.2": ["47.54", "46.43"],
            "06.1.1": ["47.73", "46.46"],
            "07.1.1": ["45.11", "45.19"],
            "07.2.1": ["45.31", "45.32"],
            "07.2.2": "47.30",
            "07.2.3": "45.20",
            "09.1.3": ["47.41", "46.51"],
            "09.1.4": "47.63",
            "09.3.1": "47.65",
            "09.3.2": "47.64",
            "09.3.3": "46.22",
            "09.5.1": "47.61",
            "09.5.2": "47.62",
        }

        konsumgrp = bef_kom["query"][0]["selection"]["values"]
        df["konsumgrp"] = konsumgrp

        # Split rows for "07.1.1"
        new_rows = []
        for _, row in df.iterrows():
            if row["konsumgrp"] in konsumgrp_mapping:
                new_value = konsumgrp_mapping[row["konsumgrp"]]
                if isinstance(new_value, list):
                    for val in new_value:
                        new_row = row.copy()
                        new_row["konsumgrp"] = val
                        new_row["value"] = row["value"] / len(
                            new_value
                        )  # Distribute value equally
                        new_rows.append(new_row)
                else:
                    row["konsumgrp"] = new_value
                    new_rows.append(row)
            else:
                new_rows.append(row)

        df = pd.DataFrame(new_rows)

        # df['konsumgrp'] = df['konsumgrp'].replace(konsumgrp_mapping)

        df["value"] = pd.to_numeric(df["value"])

        # Calculate average of "01" and "02" values
        avg_sugar_bread = (
            df[df["konsumgrp"] == "01.1.1"]["value"].iloc[0]
            + df[df["konsumgrp"] == "46.36"]["value"].iloc[0]
        ) / 2

        # Create a new DataFrame entry for the averaged value
        new_entry1 = pd.DataFrame({"konsumgrp": ["47.24"], "value": [avg_sugar_bread]})

        # Calculate average of "01" and "02" values
        avg_fruit_veges = (
            df[df["konsumgrp"] == "01.1.6"]["value"].iloc[0]
            + df[df["konsumgrp"] == "01.1.7"]["value"].iloc[0]
        ) / 2

        # Calculate average of dairy and oils
        avg_dairy = (
            df[df["konsumgrp"] == "01.1.4"]["value"].iloc[0]
            + df[df["konsumgrp"] == "01.1.5"]["value"].iloc[0]
        ) / 2

        # Create a new DataFrame entry for the averaged value
        new_entry2 = pd.DataFrame({"konsumgrp": ["47.21"], "value": [avg_fruit_veges]})

        new_entry3 = pd.DataFrame({"konsumgrp": ["46.31"], "value": [avg_fruit_veges]})

        new_entry4 = pd.DataFrame({"konsumgrp": ["46.33"], "value": [avg_dairy]})

        # Concatenate the new entry with the original DataFrame
        df = pd.concat(
            [df, new_entry1, new_entry2, new_entry3, new_entry4], ignore_index=True
        )

        # Reorder columns
        df = df[["konsumgrp", "value"]]

        return df
    else:
        print("Failed to retrieve data:", resultat1.status_code)
        return None


def process_kpi_data(year):

    # Fetch data
    kpi_total_data = fetch_kpi_data(year)
    kpi_hovedgruppe_data = fetch_hovedgruppe_data(year)
    kpi_gruppe_data = fetch_gruppe_data(year)
    subgruppe1_data = fetch_subgruppe1_data(year)

    # Concatenate the DataFrames
    kpi_df = pd.concat(
        [kpi_total_data, kpi_hovedgruppe_data, kpi_gruppe_data, subgruppe1_data],
        ignore_index=True,
    )

    # Modify the DataFrame to keep rows where 'konsumgrp' starts with 45, 46, or 47
    kpi_df = kpi_df[kpi_df["konsumgrp"].str[:2].isin(["45", "46", "47"])]

    # Sort the DataFrame by 'konsumgrp'
    kpi_df = kpi_df.sort_values(by="konsumgrp")

    # Divide 'value' by 100 to get the inflation rate
    kpi_df["value"] = kpi_df["value"] / 100 + 1

    # Rename the 'value' column to 'inflation_rate'
    kpi_df.rename(columns={"value": "inflation_rate", "konsumgrp": "n4"}, inplace=True)

    return kpi_df
