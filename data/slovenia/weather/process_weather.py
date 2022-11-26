#!/usr/bin/python3

import pandas as pd

def get_c_weather_data():
    bezigrad = pd.read_csv("bezigrad")
    bilje = pd.read_csv("bilje")
    brnik = pd.read_csv("brnik")
    celje = pd.read_csv("celje")
    cerklje = pd.read_csv("cerklje")
    maribor = pd.read_csv("maribor")
    murska_sobota = pd.read_csv("murska_sobota")
    novo_mesto = pd.read_csv("novo_mesto")
    portoroz = pd.read_csv("portoroz")
    smartno = pd.read_csv("smartno")

    stations = {
        "bezigrad": bezigrad,
        "bilje": bilje,
        "brnik": brnik,
        "celje": celje,
        "cerklje": cerklje,
        "maribor": maribor,
        "murska_sobota": murska_sobota,
        "novo_mesto": novo_mesto,
        "portoroz": portoroz,
        "smartno": smartno
    }

    # Drop columns and rename to simpler names, convert date string to datetime
    for k, v in stations.items():
        v.rename(columns=lambda x: x.strip(), inplace=True)
        v.drop(["station id", "station name"], axis=1, inplace=True)
        v.rename(columns={"valid": "date", "max. T [°C]": "max", "min. T [°C]": "min", \
            "povp. dnevna T [°C]": "avg"}, inplace=True)
        v["date"] = pd.to_datetime(v["date"], format="%Y-%m-%d")
    
    # Prefix columns with station names, merge all dataframes into one (+ remove redundant date columns)
    weather = None
    for k, v in stations.items():
        v = v.add_prefix(k + "_")
        v.rename(columns={k + "_date": "date"}, inplace=True)
        if weather is None:
            weather = v
        else:
            weather = pd.merge(weather, v, on="date", how="outer")
    
    assert weather is not None
    weather.sort_values(by=["date"], inplace=True)
    return weather

def get_avg(weather: pd.DataFrame):
    avg_mask = weather.columns.str.contains("^.*_avg$")
    weather["avg"] = weather[weather.columns[avg_mask]].mean(axis=1)
    min_mask = weather.columns.str.contains("^.*_min$")
    weather["min"] = weather[weather.columns[min_mask]].mean(axis=1)
    max_mask = weather.columns.str.contains("^.*_max$")
    weather["max"] = weather[weather.columns[max_mask]].mean(axis=1)

    avg_mask = weather.columns.str.contains("^.*_avg$")
    weather.drop(weather.columns[avg_mask], axis=1, inplace=True)
    min_mask = weather.columns.str.contains("^.*_min$")
    weather.drop(weather.columns[min_mask], axis=1, inplace=True)
    max_mask = weather.columns.str.contains("^.*_max$")
    weather.drop(weather.columns[max_mask], axis=1, inplace=True)

    return weather


# import pandas as pd
# from process_weather import get_c_weather_data