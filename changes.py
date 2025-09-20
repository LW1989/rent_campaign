def get_renter_share(renter_df):
    renter_df["Eigentuemerquote"]=(100-renter_df["Eigentuemerquote"])/100
    renter_df.rename(columns={"Eigentuemerquote":"renter_share"}, inplace=True)
    renter_df["renter_share"]=renter_df["renter_share"].round(2)

    return renter_df[["geometry", "renter_share"]]

def get_heating_type(heating_type):
    cols= ["Fernheizung", "Etagenheizung", "Blockheizung", "Zentralheizung", "Einzel_Mehrraumoefen", "keine_Heizung"]
    heating_type=calc_total(heating_type, cols)
    for col in cols:
        heating_type[col+"_share"]=heating_type[col]/heating_type["total"]

    return heating_type[["geometry"]+[c+"_share" for c in cols]]

def get_energy_type(energy_type):
    cols= ["Gas",	"Heizoel",	"Holz_Holzpellets",	"Biomasse_Biogas",	"Solar_Geothermie_Waermepumpen", "Strom", "Kohle",	"Fernwaerme", "kein_Energietraeger"]
    energy_type=calc_total(energy_type, cols)
    for col in cols:
        energy_type[col+"_share"]=energy_type[col]/energy_type["total"]
    energy_type["fossil_heating_share"]=energy_type["Gas_share"]+energy_type["Heizoel_share"]+energy_type["Kohle_share"]+energy_type["Fernwaerme_share"]
    energy_type["renewable_share"] = (energy_type["Holz_Holzpellets_share"]+energy_type["Biomasse_Biogas_share"]+energy_type["Solar_Geothermie_Waermepumpen_share"]+energy_type["Strom_share"])
    energy_type["no_energy_type"] = energy_type["kein_Energietraeger_share"]

    return energy_type[["geometry","fossil_heating_share", "renewable_share", "no_energy_type" ]]

def calc_rent_campaign_flags(
        rent_campaign_df, 
        threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6
            }
            ):

    rent_campaign_df["central_heating_flag"] = rent_campaign_df["Zentralheizung_share"] > threshold_dict["central_heating_thres"]
    rent_campaign_df["fossil_heating_flag"] = rent_campaign_df["fossil_heating_share"] > threshold_dict["fossil_heating_thres"]
    rent_campaign_df["fernwaerme_flag"] = rent_campaign_df["Fernheizung_share"] > threshold_dict["fernwaerme_thres"]
    rent_campaign_df["renter_flag"] = rent_campaign_df["renter_share"] > threshold_dict["renter_share"]

    rent_campaign_df=rent_campaign_df[rent_campaign_df["renter_flag"]==True]


    return rent_campaign_df


def get_rent_campaign_df(
        heating_type, 
        energy_type, 
        renter_df, 
        heating_typeshare_list, 
        energy_type_share_list, 
        heating_labels, 
        energy_labels,
        threshold_dict=None):
    
    if threshold_dict is None:
        threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6
            }

    logger.debug(f"Running get_heating_type with heating_type shape: {heating_type.shape}")
    heating_type_df=get_heating_type(heating_type)
    logger.debug(f"Running get_energy_type with energy_type shape: {energy_type.shape}")
    energy_type_df=get_energy_type(energy_type)
    logger.debug(f"Running get_renter_share with renter_df shape: {renter_df.shape}")
    renter_df=get_renter_share(renter_df)
   
    logger.debug(f"Merging DataFrames with shapes: heating_type_df={heating_type_df.shape}, energy_type_df={energy_type_df.shape}, renter_df={renter_df.shape}")
    rent_campaign_df=merge_dfs(
        list_of_dfs=[heating_type_df, energy_type_df, renter_df], 
        on_col="geometry", 
        how="inner")
    logger.debug(f"Resulting rent_campaign_df shape: {rent_campaign_df.shape}")

    # Calculate flags based on thresholds
    logger.debug("Calculating rent campaign flags")
    rent_campaign_df=calc_rent_campaign_flags(rent_campaign_df, threshold_dict)

    # Create pie columns
    def make_pie(row, cols, labels):
        return [
            {"label": labels[col], "value": row[col]}
            for col in cols
        ]

    # Calculate Pie Chart Values for heating and energy types
    logger.debug("Calculating pie chart values for heating and energy types")

    rent_campaign_df["heating_pie"] = rent_campaign_df.apply(
    lambda r: make_pie(r, heating_typeshare_list, heating_labels), axis=1
    )

    rent_campaign_df["energy_pie"] = rent_campaign_df.apply(
        lambda r: make_pie(r, energy_type_share_list, energy_labels), axis=1
    )

    # Drop original share columns
    rent_campaign_df = rent_campaign_df.drop(columns=heating_typeshare_list + energy_type_share_list)

    return rent_campaign_df


# This is how you call the function:

threshold_dict={
            "central_heating_thres":0.6,
            "fossil_heating_thres":0.6,
            "fernwaerme_thres":0.2,
            "renter_share":0.6
            }
renter_df=loading_dict["Zensus2022_Eigentuemerquote_100m-Gitter"].copy()
heating_type=loading_dict["Zensus2022_Gebaeude_mit_Wohnraum_nach_ueberwiegender_Heizungsart_100m-Gitter"].copy()
energy_type=loading_dict["Zensus2022_Gebaeude_mit_Wohnraum_nach_Energietraeger_der_Heizung_100m-Gitter"].copy()

# Your column groups
heating_typeshare_list = [
    "Fernheizung_share",
    "Etagenheizung_share",
    "Blockheizung_share",
    "Zentralheizung_share",
    "Einzel_Mehrraumoefen_share",
    "keine_Heizung_share",
]

energy_type_share_list = [
    "fossil_heating_share",
    "renewable_share",
    "no_energy_type",
]

# Map to labels (optional, nicer in tooltip)
heating_labels = {
    "Fernheizung_share": "Fernheizung",
    "Etagenheizung_share": "Etagenheizung",
    "Blockheizung_share": "Blockheizung",
    "Zentralheizung_share": "Zentralheizung",
    "Einzel_Mehrraumoefen_share": "Öfen",
    "keine_Heizung_share": "Keine Heizung",
}

energy_labels = {
    "fossil_heating_share": "Fossil",
    "renewable_share": "Erneuerbar",
    "no_energy_type": "Keine Angabe",
}

rent_campaign_df=get_rent_campaign_df(
        heating_type, 
        energy_type, 
        renter_df, 
        heating_typeshare_list, 
        energy_type_share_list, 
        heating_labels, 
        energy_labels,
        threshold_dict)

rent_campaign_df.head()


