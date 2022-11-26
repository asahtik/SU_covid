#!/usr/bin/python3

import pandas as pd

def merge_station_data():
    stations = pd.read_csv("station_metadata.csv")
    data = pd.read_csv("hourly_temperatures.csv")

    stations = stations[["id", "Bundesland"]]
    stations.rename(columns={"id": "station"}, inplace=True)

    data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S")


    data = pd.merge(data, stations, on="station", how="left")

    data["Bundesland"] = data["Bundesland"].map({"NOE": "Niederösterreich", "STMK": "Steiermark", "OOE": "Oberösterreich", "WIE": "Wien"})

    data.rename(columns={
        "Bundesland": "state",
        "LT2": "T5cm",
        "TTX": "T2m"
    }, inplace=True)

    return data

def to_daily(data):
    data["date"] = data["time"].dt.date
    data.drop("time", axis=1, inplace=True)
    data["temperature"] = data[["T5cm", "T2m"]].mean(axis=1)
    data.drop(["T5cm", "T2m"], axis=1, inplace=True)
    grouped = data.groupby(["date", "station", "state"]).\
        agg(Mean=("temperature", "mean"), Min=("temperature", "min"), Max=("temperature", "max")).reset_index()
    grouped.drop("station", axis=1, inplace=True)
    grouped = grouped.groupby(["date", "state"]).mean().reset_index()

    grouped.rename(columns={
        "Mean": "avg",
        "Min": "min",
        "Max": "max"
    }, inplace=True)

    return grouped
