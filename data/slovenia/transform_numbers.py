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

    stats["date"] = pd.to_datetime(stats["date"], format="%Y-%m-%d")

    return stats