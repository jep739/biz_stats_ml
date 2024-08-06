def networks(aar):
    import sgis as sg

    url = "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    veger = sg.read_parquet_url(url)
    veger

    regler = sg.NetworkAnalysisRules(
        weight="minutes",
        directed=True,
        search_tolerance=500,  # meter
    )

    antall_per_retning = veger.oneway.value_counts().to_frame()

    antall_per_retning["hva_må_gjøres"] = {
        "B": "Duplisere og snu duplikatene",
        "TF": "Snu",
        "FT": "Ingenting",
    }

    veger[["oneway", "drivetime_fw", "drivetime_bw"]]

    veger_med_retning = sg.make_directed_network_norway(veger, dropnegative=True)

    veger_ned_tilknytning = sg.get_connected_components(veger_med_retning).query(
        "connected == 1"
    )

    nwa = sg.NetworkAnalysis(network=veger_ned_tilknytning, rules=regler)

    vippetangen = sg.to_gdf([10.741527, 59.9040595], crs=4326).to_crs(veger.crs)
    ryen = sg.to_gdf([10.8047522, 59.8949826], crs=4326).to_crs(veger.crs)

    url = "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    punkter = sg.read_parquet_url(url)[["geometry"]]

    #     bygningspunktsti = f"ssb-prod-kart-data-delt/kartdata_analyse/klargjorte-data/2022/SMAT_bygg_punkt_p2022_v1.parquet"

    #     bygg = sg.read_geopandas(
    #         bygningspunktsti,
    #         filters=[("KOMMUNENR", "in",  ("0301", "3024", "3020"))],
    #         columns=["geometry"]
    #     )

    #     punkter = bygg.sample(1000).reset_index(drop=True)

    frequencies = nwa.get_route_frequencies(
        origins=punkter.sample(100),
        destinations=punkter.sample(100),
    )

    frequencies.nlargest(3, "frequency")

    return frequencies
