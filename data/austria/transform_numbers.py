#!/usr/bin/python3

import pandas as pd

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
    numbers.to_csv("covid_numbers_per_state.csv", index=False)

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
    hospitals.to_csv("hospital_numbers_per_state.csv", index=False)

# TODO: remove states with population < 1,000,000 !(Niederösterreich, Steiermark, Oberösterreich, Wien), merge covid and hospital numbers,
# merge
transform_hospital_numbers()