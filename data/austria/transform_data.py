#!/usr/bin/python3

import pandas as pd

states = ["Niederösterreich", "Steiermark", "Oberösterreich", "Wien"]

def transform_covid_numbers():
    numbers = pd.read_csv("CovidFaelle_Altersgruppe.csv", delimiter=";")
    numbers.drop(["AltersgruppeID", "Altersgruppe", "BundeslandID", "Geschlecht"], axis=1, inplace=True)
    numbers["Time"] = pd.to_datetime(numbers["Time"], format="%d.%m.%Y %H:%M:%S")
    numbers = numbers.groupby(["Time", "Bundesland"]).sum().reset_index()
    numbers.rename(columns={
        "Bundesland": "state", 
        "Time": "date", 
        "AnzEinwohner": "population", 
        "Anzahl": "cases",
        "AnzahlGeheilt": "recovered",
        "AnzahlTot": "deceased"
        }, inplace=True)
    numbers.sort_values(by=["date"], inplace=True)
    return numbers

def transform_hospital_numbers():
    hospitals = pd.read_csv("CovidFallzahlen.csv", delimiter=";")
    hospitals.drop(["MeldeDatum", "BundeslandID", "FZHospFree", "FZICUFree"], axis=1, inplace=True)
    hospitals["Meldedat"] = pd.to_datetime(hospitals["Meldedat"], format="%d.%m.%Y")
    hospitals.rename(columns={
        "Meldedat": "date",
        "TestGesamt": "tests", 
        "FZHosp": "normal_beds",
        "FZICU": "ICU_beds",
        "Bundesland": "state"
        }, inplace=True)
    hospitals.sort_values(by=["date"], inplace=True)
    return hospitals

def add_features(data):
    holidays = [
        (1, 1),
        (6, 1),
        (1, 5),
        (26, 5),
        (6, 6),
        (15, 8),
        (26, 10),
        (1, 11),
        (8, 12),
        (24, 12),
        (25, 12),
        (26, 12),
        (31, 12)
    ]
    easters = [pd.Timestamp("2020-04-12"), pd.Timestamp("2021-04-04"), pd.Timestamp("2022-04-17")]

    start_date = pd.Timestamp("2020-02-25") # Day after patient zero
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data["day"] = (data["date"] - start_date).dt.days
    data["dow"] = data["date"].dt.day_name("en_US.utf8")
    data["holiday"] = data["date"].apply(lambda x: (x.day, x.month) in holidays or x in easters)

    q1 = pd.Timestamp("2020-03-16")
    q2 = pd.Timestamp("2020-04-21")
    q3 = pd.Timestamp("2020-11-17")
    q4 = pd.Timestamp("2020-12-07")
    q5 = pd.Timestamp("2020-12-26")
    q6 = pd.Timestamp("2021-02-08")
    q7 = pd.Timestamp("2021-11-22")
    q8 = pd.Timestamp("2021-12-13")
    data["lockdown"] = (q1 <= data["date"]) & (data["date"] < q2) | (q3 <= data["date"]) & (data["date"] < q4) | \
        (q5 <= data["date"]) & (data["date"] < q6) | (q7 <= data["date"]) & (data["date"] < q8)
    return data

def transform():
    covid = transform_covid_numbers()
    hospitals = transform_hospital_numbers()
    covid = covid[covid["state"].isin(states)]
    hospitals = hospitals[hospitals["state"].isin(states)]
    merged = pd.merge(covid, hospitals, on=["date", "state"], how="outer")
    merged.sort_values(by=["date", "state"], inplace=True)

    weather = pd.read_csv("weather/weather_austria.csv")
    weather.rename(columns={
        "avg": "temp_avg",
        "min": "temp_min",
        "max": "temp_max"
        }, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"], format="%Y-%m-%d")
    merged = pd.merge(merged, weather, on=["date", "state"], how="left")
    return add_features(merged)

# def merge_weather():
#     stats = pd.read_csv("covid_data_austria.csv")
#     weather = pd.read_csv("weather/weather_austria.csv")
#     weather.rename(columns={
#         "avg": "temp_avg",
#         "min": "temp_min",
#         "max": "temp_max"
#         }, inplace=True)
#     return pd.merge(stats, weather, on=["date", "state"], how="left")