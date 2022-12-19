#!/usr/bin/python3

import pandas as pd

def add_features(data):
    holidays = [
        (1, 1), (2, 1),
        (8, 2),
        (27, 4),
        (1, 5),
        (2, 5),
        (25, 6),
        (15, 8),
        (31, 10),
        (1, 11),
        (25, 12),
        (26, 12)
    ]
    easters = [pd.Timestamp("2020-04-12"), pd.Timestamp("2021-04-04"), pd.Timestamp("2022-04-17")]

    start_date = pd.Timestamp("2020-03-05") # Day after patient zero
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data["day"] = (data["date"] - start_date).dt.days
    data["dow"] = data["date"].dt.day_name("en_US.utf8")
    q1 = pd.Timestamp("2020-03-12")
    q2 = pd.Timestamp("2020-05-15")
    q3 = pd.Timestamp("2020-10-19")
    q4 = pd.Timestamp("2020-12-03")
    data["lockdown"] = (q1 <= data["date"]) & (data["date"] < q2) | (q3 <= data["date"]) & (data["date"] < q4)
    data["holiday"] = data["date"].apply(lambda x: (x.day, x.month) in holidays or x in easters)
    return data

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

    weather = pd.read_csv("weather/weather_slovenia.csv")
    weather.rename(columns={
        "avg": "temp_avg",
        "min": "temp_min",
        "max": "temp_max"
        }, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"], format="%Y-%m-%d")
    stats = pd.merge(stats, weather, on=["date"], how="left")
    return add_features(stats)

# def merge_weather():
#     stats = pd.read_csv("covid_data_slovenia.csv")
#     weather = pd.read_csv("weather/weather_slovenia.csv")
#     weather.rename(columns={
#         "avg": "temp_avg",
#         "min": "temp_min",
#         "max": "temp_max"
#         }, inplace=True)
#     return pd.merge(stats, weather, on=["date"], how="left")
