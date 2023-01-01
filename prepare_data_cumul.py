import numpy as np
import pandas as pd

PER_100000 = True

dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
slovenia = pd.read_csv("data/slovenia/data_slovenia.csv")
austria = pd.read_csv("data/austria/data_austria.csv")

austria.sort_values(by=["date"], inplace=True)

slovenia["dow"] = slovenia["dow"].map(dow_map)
slovenia["dow_cos"] = np.cos(slovenia["dow"] * (2 * np.pi / 7))
slovenia["dow_sin"] = np.sin(slovenia["dow"] * (2 * np.pi / 7))
slovenia.drop("dow", axis=1, inplace=True)

austria["dow"] = austria["dow"].map(dow_map)
austria["dow_cos"] = np.cos(austria["dow"] * (2 * np.pi / 7))
austria["dow_sin"] = np.sin(austria["dow"] * (2 * np.pi / 7))
austria.drop("dow", axis=1, inplace=True)

slovenia.drop("date", axis=1, inplace=True)
austria.drop("date", axis=1, inplace=True)

slovenia.dropna(inplace=True)
austria.dropna(inplace=True)
slovenia.reset_index(drop=True, inplace=True)
austria.reset_index(drop=True, inplace=True)

states = {}
slovenia["population"] = 2105000
states["Slovenia"] = slovenia
grouped = austria.groupby("state")
for state in austria["state"].unique():
    states[state] = grouped.get_group(state)
    states[state] = states[state].drop("state", axis=1)

for _, v in states.items():
    # Calculate per capita
    if PER_100000:
        v["ICU_beds"] = v["ICU_beds"].cumsum() / v["population"] * 100000
        v["cases"] = v["cases"].cumsum() / v["population"] * 100000
        v["deceased"] = v["deceased"].cumsum() / v["population"] * 100000
        v["normal_beds"] = v["normal_beds"].cumsum() / v["population"] * 100000
        v["recovered"] = v["recovered"].cumsum() / v["population"] * 100000
        v["tests"] = v["tests"].cumsum() / v["population"] * 100000
    else:
        v["ICU_beds"] = v["ICU_beds"].cumsum() / v["population"]
        v["cases"] = v["cases"].cumsum() / v["population"]
        v["deceased"] = v["deceased"].cumsum() / v["population"]
        v["normal_beds"] = v["normal_beds"].cumsum() / v["population"]
        v["recovered"] = v["recovered"].cumsum() / v["population"]
        v["tests"] = v["tests"].cumsum() / v["population"]

    v.sort_index(axis=1, inplace=True)

    # Normalise. A model should not have access to future data, 
    # however I don't have time to overcomplicate
    v["day"] = (v["day"] - v["day"].mean()) / v["day"].std()
    # v["ICU_beds_per_100k"] = (v["ICU_beds_per_100k"] - v["ICU_beds_per_100k"].mean()) / v["ICU_beds_per_100k"].std()
    # v["cases_per_100k"] = (v["cases_per_100k"] - v["cases_per_100k"].mean()) / v["cases_per_100k"].std()
    # v["deceased_per_100k"] = (v["deceased_per_100k"] - v["deceased_per_100k"].mean()) / v["deceased_per_100k"].std()
    # v["normal_beds_per_100k"] = (v["normal_beds_per_100k"] - v["normal_beds_per_100k"].mean()) / v["normal_beds_per_100k"].std()
    # v["recovered_per_100k"] = (v["recovered_per_100k"] - v["recovered_per_100k"].mean()) / v["recovered_per_100k"].std()
    # v["tests_per_100k"] = (v["tests_per_100k"] - v["tests_per_100k"].mean()) / v["tests_per_100k"].std()
    v["temp_avg"] = (v["temp_avg"] - v["temp_avg"].mean()) / v["temp_avg"].std()
    v["temp_max"] = (v["temp_max"] - v["temp_max"].mean()) / v["temp_max"].std()
    v["temp_min"] = (v["temp_min"] - v["temp_min"].mean()) / v["temp_min"].std()

for k, v in states.items():
    v.to_csv("data/processed/pn_cumul_data_" + k[0:3].lower() + ".csv", index=False)