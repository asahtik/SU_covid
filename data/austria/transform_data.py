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

def transform():
    covid = transform_covid_numbers()
    hospitals = transform_hospital_numbers()
    covid = covid[covid["state"].isin(states)]
    hospitals = hospitals[hospitals["state"].isin(states)]
    merged = pd.merge(covid, hospitals, on=["date", "state"], how="outer")
    merged.sort_values(by=["date", "state"], inplace=True)
    return merged

def merge_weather():
    stats = pd.read_csv("covid_data_austria.csv")
    weather = pd.read_csv("weather/weather_austria.csv")
    weather.rename(columns={
        "avg": "temp_avg",
        "min": "temp_min",
        "max": "temp_min"
        }, inplace=True)
    return pd.merge(stats, weather, on=["date", "state"], how="left")