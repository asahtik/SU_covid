#!/usr/bin/python3
import random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import models
from keras import layers

from WindowGenerator import WindowGenerator

INPUT_SIZE = 30
PREDICT_SIZE = 7
TOTAL_SIZE = INPUT_SIZE + PREDICT_SIZE
OVERLAP = 7
TEST_PERCENTAGE = 0.2

SEED = 42

BATCH_SIZE = 8

rnd.seed(SEED)

features = ["ICU_beds_per_100k", "cases_per_100k", "day", "deceased_per_100k",
            "dow_cos", "dow_sin", "holiday", "lockdown", "normal_beds_per_100k",
            "recovered_per_100k", "temp_avg", "temp_max", "temp_min",
            "tests_per_100k"]
# labels = ["cases_per_100k", "normal_beds_per_100k", "ICU_beds_per_100k", "deceased_per_100k"]
labels = ["cases_per_100k"]

def compile_and_fit(model, train_ds, val_ds, patience=2, max_epochs=20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(train_ds, epochs=max_epochs,
                      validation_data=val_ds,
                      callbacks=[early_stopping])
  return history

# def plot(title, values, xcol, ycol, multi=False):
#     plt.figure()
#     if multi:
#         for v in values:
#             plt.scatter(v[xcol], v[ycol])
#     else:
#         plt.scatter(values[xcol], values[ycol])
#     plt.title(title)
#     plt.show()

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

# plot("Slovenia", states["Slovenia"], "day", "cases", False)

# for c in states["Slovenia"].columns:
#     plt.title(c)
#     plt.scatter(range(0, len(states["Slovenia"])), states["Slovenia"][c])
#     plt.show()
#     plt.clf()
#     plt.title(c)
#     plt.scatter(range(0, len(states["Wien"])), states["Wien"][c])
#     plt.show()
#     plt.clf()

for _, v in states.items():
    # Calculate per capita
    v["ICU_beds_per_100k"] = v["ICU_beds"] / v["population"] * 100000
    v["cases_per_100k"] = v["cases"] / v["population"] * 100000
    v["deceased_per_100k"] = v["deceased"] / v["population"] * 100000
    v["normal_beds_per_100k"] = v["normal_beds"] / v["population"] * 100000
    v["recovered_per_100k"] = v["recovered"] / v["population"] * 100000
    v["tests_per_100k"] = v["tests"] / v["population"] * 100000
    v.drop(["ICU_beds", "population", "cases", "deceased", "normal_beds", "recovered", "tests"], axis=1, inplace=True)
    v.sort_index(axis=1, inplace=True)

    # Normalise. A model should not have access to future data, 
    # however I don't have time to overcomplicate
    v["day"] = (v["day"] - v["day"].mean()) / v["day"].std()
    v["ICU_beds_per_100k"] = (v["ICU_beds_per_100k"] - v["ICU_beds_per_100k"].mean()) / v["ICU_beds_per_100k"].std()
    v["cases_per_100k"] = (v["cases_per_100k"] - v["cases_per_100k"].mean()) / v["cases_per_100k"].std()
    v["deceased_per_100k"] = (v["deceased_per_100k"] - v["deceased_per_100k"].mean()) / v["deceased_per_100k"].std()
    v["normal_beds_per_100k"] = (v["normal_beds_per_100k"] - v["normal_beds_per_100k"].mean()) / v["normal_beds_per_100k"].std()
    v["recovered_per_100k"] = (v["recovered_per_100k"] - v["recovered_per_100k"].mean()) / v["recovered_per_100k"].std()
    v["tests_per_100k"] = (v["tests_per_100k"] - v["tests_per_100k"].mean()) / v["tests_per_100k"].std()
    v["temp_avg"] = (v["temp_avg"] - v["temp_avg"].mean()) / v["temp_avg"].std()
    v["temp_max"] = (v["temp_max"] - v["temp_max"].mean()) / v["temp_max"].std()
    v["temp_min"] = (v["temp_min"] - v["temp_min"].mean()) / v["temp_min"].std()

w = WindowGenerator(INPUT_SIZE, PREDICT_SIZE, 1, OVERLAP, states["Slovenia"].columns, label_columns=labels)

def merge_segments(segments):
    ret = []
    for i in range(len(segments)):
        if i == 0:
            ret.append(segments[i][1])
        elif segments[i][0] - segments[i - 1][0] == 1:
            ret[-1] = pd.concat([ret[-1], segments[i][1]])
        else:
            ret.append(segments[i][1])
    return ret

test_ds = {}
train_ds = {}
# Split into train and test
for k, v in states.items():
    l = len(v)
    seg_size = TOTAL_SIZE
    num_segments = l // seg_size
    all_s = np.arange(num_segments)
    test_s = np.array(rnd.sample(list(all_s), round(num_segments * TEST_PERCENTAGE)))
    test_t = []
    train_t = []
    for i in range(num_segments):
        if i in test_s:
            test_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
        else:
            train_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
    test_ds[k] = merge_segments(test_t)
    train_ds[k] = merge_segments(train_t)
    # plot(k, train_ds[k], "day", "cases_per_100k", True)

tf_train_ds = w.make_dataset(train_ds, BATCH_SIZE)
tf_test_ds = w.make_dataset(test_ds, 1)

print(tf_train_ds)

print("Dataset done")

model = models.Sequential([
    layers.Dense(32),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(len(labels) * PREDICT_SIZE),
    layers.Reshape([PREDICT_SIZE, len(labels)])
])

h = compile_and_fit(model, tf_train_ds, tf_test_ds)

d = tf_test_ds.skip(2).take(1).get_single_element()
pred = model(d[0])
plt.scatter(range(0, len(d[0][0, :, 0])), d[0][0, :, 0])
iend = len(d[0][0, :, 0])
plt.scatter(range(iend, iend + len(d[1][0, :, 0])), d[1][0, :, 0], label="True")
plt.scatter(range(iend, iend + len(pred[0, :, 0])), pred[0, :, 0], label="Predicted")
plt.legend()
plt.show()

print("A")
print("B")