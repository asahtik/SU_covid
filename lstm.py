#!/usr/bin/python3
import pandas as pd

from keras import models
from keras import layers

from helpers import get_data, get_data_loo, get_data_kcv, compile_and_fit, compile, fit, test, get_data2

INPUT_SIZE = 30
PREDICT_SIZE = 7
VALIDATION_PERCENTAGE = 0.3

MAX_EPOCHS = 100
PATIENCE = 10

SEED = 42

BATCH_SIZE = 16

features = ["ICU_beds", "cases", "day", "deceased",
            "dow_cos", "dow_sin", "holiday", "lockdown", "normal_beds",
            "recovered", "temp_avg", "temp_max", "temp_min",
            "tests"]
# ifeatures = ["cases", "day"]
ifeatures = ["cases", "day", "dow_cos", "dow_sin", "holiday", "lockdown", "temp_avg"]
# labels = ["cases_per_100k", "normal_beds_per_100k", "ICU_beds_per_100k", "deceased_per_100k"]
labels = ["cases"]


# tf_test_ds, tf_train_ds, tf_val_ds = get_data(INPUT_SIZE, PREDICT_SIZE, features, labels, TEST_PERCENTAGE, VALIDATION_PERCENTAGE, BATCH_SIZE)

# mlp = models.Sequential([
#     layers.Input(shape=(INPUT_SIZE, len(ifeatures))),
#     layers.Flatten(),
#     # layers.Dense(64, activation="relu"),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(len(labels) * PREDICT_SIZE, activation="relu"),
#     layers.Reshape([PREDICT_SIZE, len(labels)])
# ])

# lstm = models.Sequential([
#     layers.Input(shape=(INPUT_SIZE, len(ifeatures))),
#     # layers.LSTM(64, return_sequences=True, activation="relu"),
#     layers.LSTM(128, activation="relu"),
#     layers.Dense(len(labels) * PREDICT_SIZE, activation="relu"),
#     layers.Reshape([PREDICT_SIZE, len(labels)])
# ])

# gru = models.Sequential([
#     layers.Input(shape=(INPUT_SIZE, len(ifeatures))),
#     # layers.GRU(64, return_sequences=True, activation="relu"),
#     layers.GRU(128, activation="relu"),
#     layers.Dense(len(labels) * PREDICT_SIZE, activation="relu"),
#     layers.Reshape([PREDICT_SIZE, len(labels)])
# ])

# compile(mlp)
# mlp_initial_weights = mlp.get_weights()
# compile(lstm)
# lstm_initial_weights = lstm.get_weights()
# compile(gru)
# gru_initial_weights = gru.get_weights()

# iter = 0
# results = pd.DataFrame(columns=["mlp_loss", "mlp_mae", "lstm_loss", "lstm_mae", "gru_loss", "gru_mae"])
# for (test_countries, test_indices), (tf_test_ds, tf_train_ds, tf_val_ds) in get_data_kcv(10, INPUT_SIZE, PREDICT_SIZE, ifeatures, labels, VALIDATION_PERCENTAGE, BATCH_SIZE):
#     iter += 1
#     print("Iteration %d" % (iter))
#     mlp.set_weights(mlp_initial_weights)
#     lstm.set_weights(lstm_initial_weights)
#     gru.set_weights(gru_initial_weights)
#     print("Fit")
#     hmlp = fit(mlp, tf_train_ds, tf_val_ds, patience=PATIENCE, max_epochs=MAX_EPOCHS)
#     # hlstm = fit(lstm, tf_train_ds, tf_val_ds, patience=PATIENCE, max_epochs=MAX_EPOCHS)
#     hgru = fit(gru, tf_train_ds, tf_val_ds, patience=PATIENCE, max_epochs=MAX_EPOCHS)
#     print("Test")
#     resmlp = mlp.evaluate(tf_test_ds)
#     reslstm = (0, 0)# reslstm = lstm.evaluate(tf_test_ds)
#     resgru = gru.evaluate(tf_test_ds)
#     tres = pd.DataFrame(
#         columns=["mlp_loss", "mlp_mae", "lstm_loss", "lstm_mae", "gru_loss", "gru_mae"],
#         data=[[resmlp[0], resmlp[1], reslstm[0], reslstm[1], resgru[0], resgru[1]]]
#     )
#     results = pd.concat([results, tres])

# results.to_csv("results.csv")

(tf_test_ds1, tf_train_ds1, tf_val_ds1), (tf_test_ds2, tf_train_ds2, tf_val_ds2) = get_data2(INPUT_SIZE, PREDICT_SIZE, ["cases", "day"], 
["cases", "day", "dow_cos", "dow_sin", "holiday", "lockdown", "temp_avg"], labels, 0.1, VALIDATION_PERCENTAGE, BATCH_SIZE)

gru1 = models.Sequential([
    layers.Input(shape=(INPUT_SIZE, 2)),
    # layers.GRU(64, return_sequences=True, activation="relu"),
    layers.GRU(64, activation="relu"),
    layers.Dense(len(labels) * PREDICT_SIZE, activation="relu"),
    layers.Reshape([PREDICT_SIZE, len(labels)])
])
gru2 = models.Sequential([
    layers.Input(shape=(INPUT_SIZE, 7)),
    # layers.GRU(64, return_sequences=True, activation="relu"),
    layers.GRU(64, activation="relu"),
    layers.Dense(len(labels) * PREDICT_SIZE, activation="relu"),
    layers.Reshape([PREDICT_SIZE, len(labels)])
])
compile(gru1)
compile(gru2)

hgru1 = fit(gru1, tf_train_ds1, tf_val_ds1, patience=100, max_epochs=1000, start_from_epoch=100)
hgru2 = fit(gru2, tf_train_ds2, tf_val_ds2, patience=100, max_epochs=1000, start_from_epoch=100)
gru1.save("tf_saved/grub")
gru2.save("tf_saved/grue")
tf_test_ds1.save("tf_saved/grub_test")
tf_test_ds2.save("tf_saved/grue_test")