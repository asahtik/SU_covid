#!/usr/bin/python3

import pandas as pd

def transform():
    stats = pd.read_csv("stats.csv")

    columns = ["date", "tests.performed", "cases.active", "state.in_hospital", "state.icu", "state.deceased.todate", "cases.recovered.todate"]

    stats = stats[columns]
    
    stats.rename(columns={
        "tests.performed": "tests",
        "cases.active": "cases",
        "state.in_hospital": "normal_beds",
        "state.icu": "ICU_beds",
        "state.deceased.todate": "deceased",
        "cases.recovered.todate": "recovered"
    }, inplace=True)

    stats["normal_beds"] = stats["normal_beds"] - stats["ICU_beds"]
    stats["date"] = pd.to_datetime(stats["date"], format="%Y-%m-%d")
    stats.sort_values(by=["date"], inplace=True)
    stats["deceased"] = stats["deceased"].diff()
    stats["recovered"] = stats["recovered"].diff()

    return stats

def merge_weather():
    stats = pd.read_csv("covid_data_slovenia.csv")
    weather = pd.read_csv("weather/weather_slovenia.csv")
    weather.rename(columns={
        "avg": "temp_avg",
        "min": "temp_min",
        "max": "temp_min"
        }, inplace=True)
    return pd.merge(stats, weather, on=["date"], how="left")