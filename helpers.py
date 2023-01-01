import random as rnd
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from WindowGenerator import WindowGenerator

SEED = 42

rnd.seed(SEED)

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

def get_data(input_size, predict_size, features, labels, test_percentage, validation_percentage, batch_size):
    states = {
        "Slovenia": pd.read_csv("data/processed/pn_data_slo.csv"),
        "Niederösterreich": pd.read_csv("data/processed/pn_data_nie.csv"),
        "Oberösterreich": pd.read_csv("data/processed/pn_data_obe.csv"),
        "Steiermark": pd.read_csv("data/processed/pn_data_ste.csv"),
        "Wien": pd.read_csv("data/processed/pn_data_wie.csv")
    }

    print(states["Slovenia"].columns)
    print(states["Wien"].columns)
    w = WindowGenerator(input_size, predict_size, 1, states["Slovenia"].columns, feature_columns=features, label_columns=labels)

    test_ds = {}
    train_ds = {}
    val_ds = {}
    # Split into train, test, validation
    for k, v in states.items():
        l = len(v)
        seg_size = input_size + predict_size
        num_segments = l // seg_size
        all_s = np.arange(num_segments)
        test_s = np.array(rnd.sample(list(all_s), round(num_segments * test_percentage)))
        inter_s = np.setdiff1d(all_s, test_s)
        val_s = np.array(rnd.sample(list(inter_s), round(num_segments * validation_percentage)))
        test_t = []
        train_t = []
        val_t = []
        for i in range(num_segments):
            if i in test_s:
                test_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
            elif i in val_s:
                val_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
            else:
                train_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
        test_ds[k] = merge_segments(test_t)
        train_ds[k] = merge_segments(train_t)
        val_ds[k] = merge_segments(val_t)
    return (w.make_dataset(test_ds, 1), w.make_dataset(train_ds, batch_size), w.make_dataset(val_ds, 1))

def get_data_loo(input_size, predict_size, features, labels, validation_percentage, batch_size):
    states = {
        "Slovenia": pd.read_csv("data/processed/pn_cumul_data_slo.csv"),
        "Niederösterreich": pd.read_csv("data/processed/pn_cumul_data_nie.csv"),
        "Oberösterreich": pd.read_csv("data/processed/pn_cumul_data_obe.csv"),
        "Steiermark": pd.read_csv("data/processed/pn_cumul_data_ste.csv"),
        "Wien": pd.read_csv("data/processed/pn_cumul_data_wie.csv")
    }

    w = WindowGenerator(input_size, predict_size, 1, states["Slovenia"].columns, feature_columns=features, label_columns=labels)

    total_num_segments = 0
    for _, v in states.items():
        l = len(v)
        seg_size = input_size + predict_size
        num_segments = l // seg_size
        total_num_segments += num_segments

    for t in range(0, total_num_segments):
        test_ds = {}
        train_ds = {}
        val_ds = {}
        cumul_num_segments = 0
        test_country = ""
        test_index = -1
        for k, v in states.items():
            train_t = []
            val_t = []
            l = len(v)
            seg_size = w.total_window_size
            num_segments = l // seg_size
            all_s = np.arange(num_segments)
            if cumul_num_segments <= t and t < cumul_num_segments + num_segments:
                ind = t - cumul_num_segments
                test_ds[k] = [v.iloc[ind * seg_size:(ind + 1) * seg_size].astype(np.float32)]
                all_s = np.delete(all_s, ind)
                test_country = k
                test_index = ind * seg_size
            cumul_num_segments += num_segments
            val_s = np.array(rnd.sample(list(all_s), round(num_segments * validation_percentage)))
            for i in range(num_segments):
                if i in val_s:
                    val_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
                else:
                    train_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
            train_ds[k] = merge_segments(train_t)
            val_ds[k] = merge_segments(val_t)
        yield (test_country, test_index), (w.make_dataset(test_ds, 1), w.make_dataset(train_ds, batch_size), w.make_dataset(val_ds, 1))

def get_data_kcv(k, input_size, predict_size, features, labels, validation_percentage, batch_size):
    assert k > 1

    states = {
        "Slovenia": pd.read_csv("data/processed/pn_cumul_data_slo.csv"),
        "Niederösterreich": pd.read_csv("data/processed/pn_cumul_data_nie.csv"),
        "Oberösterreich": pd.read_csv("data/processed/pn_cumul_data_obe.csv"),
        "Steiermark": pd.read_csv("data/processed/pn_cumul_data_ste.csv"),
        "Wien": pd.read_csv("data/processed/pn_cumul_data_wie.csv")
    }

    w = WindowGenerator(input_size, predict_size, 0, states["Slovenia"].columns, feature_columns=features, label_columns=labels)

    total_num_segments = 0
    for _, v in states.items():
        l = len(v)
        seg_size = input_size + predict_size
        num_segments = l // seg_size
        total_num_segments += num_segments

    max_test_segments = round(total_num_segments / k)

    for t in range(0, k):
        test_ds = {}
        train_ds = {}
        val_ds = {}
        k_start = t * max_test_segments
        k_end = (t + 1) * max_test_segments
        cumul_num_segments = 0
        test_countries = []
        test_indices = []
        for k, v in states.items():
            test_t = []
            train_t = []
            val_t = []
            l = len(v)
            seg_size = w.total_window_size
            num_segments = l // seg_size
            all_s = np.arange(num_segments)
            test_s = np.array([])
            if cumul_num_segments <= k_start and k_start < cumul_num_segments + num_segments:
                ind_start = k_start - cumul_num_segments
                ind_end = min(len(all_s), k_end - cumul_num_segments)
                test_s =  np.array(range(ind_start, ind_end))
                all_s = np.delete(all_s, range(ind_start, ind_end))
                test_countries.extend([k for _ in range(ind_start * seg_size, ind_end * seg_size)])
                test_indices.extend(range(ind_start * seg_size, ind_end * seg_size))
            elif cumul_num_segments <= k_end and k_end < cumul_num_segments + num_segments:
                ind_start = max(0, k_start - cumul_num_segments)
                ind_end = k_end - cumul_num_segments
                test_s =  np.array(range(ind_start, ind_end))
                all_s = np.delete(all_s, range(ind_start, ind_end))
                test_countries.extend([k for _ in range(ind_start, ind_end)])
                test_indices.extend(range(ind_start, ind_end))
            cumul_num_segments += num_segments
            val_s = np.array(rnd.sample(list(all_s), round(num_segments * validation_percentage)))
            for i in range(num_segments):
                if i in test_s:
                    test_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
                elif i in val_s:
                    val_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
                else:
                    train_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
            if len(test_t) > 0:
                test_ds[k] = merge_segments(test_t)
            if len(train_t) > 0:
                train_ds[k] = merge_segments(train_t)
            if len(val_t) > 0:
                val_ds[k] = merge_segments(val_t)
        yield (test_countries, test_indices), (w.make_dataset(test_ds, batch_size, shuffle=False), w.make_dataset(train_ds, batch_size), w.make_dataset(val_ds, batch_size))

def compile(model):
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

def fit(model, train_ds, val_ds, patience=2, max_epochs=20, start_from_epoch=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      restore_best_weights=True,
                                                      verbose=1,
                                                      start_from_epoch=start_from_epoch)
    history = model.fit(train_ds, epochs=max_epochs,
                        validation_data=val_ds,
                        callbacks=[early_stopping])
    return history

def compile_and_fit(model, train_ds, val_ds, patience=2, max_epochs=20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    restore_best_weights=True,
                                                    start_from_epoch=10)
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(train_ds, epochs=max_epochs,
                      validation_data=val_ds,
                      callbacks=[early_stopping])
  return history

def test(model, data, iter, plot=False, **kwargs) -> np.ndarray | None:
    pred_size = data.take(1).element_spec[1].shape[1]
    labels = data.take(1).element_spec[1].shape[2]
    mean_errors = np.zeros((pred_size, labels))
    num_tests = 0
    for i, o in data:
        num_tests += 1
        pred = model(i) # [batches (1), pred_size, labels]
        if plot:
            plt.figure(figsize=(2 * labels, 8))
            for j in range(0, labels):
                axes = plt.subplot(labels, 1, j + 1)
                axes.scatter(range(0, pred_size), pred[0, :, j], label="Predicted")
                axes.scatter(range(0, pred_size), o[0, :, j], label="True")
                axes.legend()
            plt.title("Iteration %d, test %d" % (iter, num_tests))
            prefix = kwargs.get("fileprefix", "")
            plt.savefig("%s_i%dt%d.png" % (prefix, iter, num_tests))
            plt.close()
        for j in range(pred_size):
            for k in range(labels):
                mean_errors[j, k] += np.abs(pred[0, j, k] - o[0, j, k])
    mean_errors = mean_errors / num_tests if num_tests > 0 else None
    return mean_errors

def get_data2(input_size, predict_size, features1, features2, labels, test_percentage, validation_percentage, batch_size):
    states = {
        "Slovenia": pd.read_csv("data/processed/pn_cumul_data_slo.csv"),
        "Niederösterreich": pd.read_csv("data/processed/pn_cumul_data_nie.csv"),
        "Oberösterreich": pd.read_csv("data/processed/pn_cumul_data_obe.csv"),
        "Steiermark": pd.read_csv("data/processed/pn_cumul_data_ste.csv"),
        "Wien": pd.read_csv("data/processed/pn_cumul_data_wie.csv")
    }

    print(states["Slovenia"].columns)
    print(states["Wien"].columns)
    w1 = WindowGenerator(input_size, predict_size, 1, states["Slovenia"].columns, feature_columns=features1, label_columns=labels)
    w2 = WindowGenerator(input_size, predict_size, 1, states["Slovenia"].columns, feature_columns=features2, label_columns=labels)

    test_ds = {}
    train_ds = {}
    val_ds = {}
    # Split into train, test, validation
    for k, v in states.items():
        l = len(v)
        seg_size = input_size + predict_size
        num_segments = l // seg_size
        all_s = np.arange(num_segments)
        test_s = np.array(rnd.sample(list(all_s), round(num_segments * test_percentage)))
        inter_s = np.setdiff1d(all_s, test_s)
        val_s = np.array(rnd.sample(list(inter_s), round(num_segments * validation_percentage)))
        test_t = []
        train_t = []
        val_t = []
        for i in range(num_segments):
            if i in test_s:
                test_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
            elif i in val_s:
                val_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
            else:
                train_t.append((i, v.iloc[i * seg_size:(i + 1) * seg_size].astype(np.float32)))
        test_ds[k] = merge_segments(test_t)
        train_ds[k] = merge_segments(train_t)
        val_ds[k] = merge_segments(val_t)
    
    d1 = (w1.make_dataset(test_ds, 1), w1.make_dataset(train_ds, batch_size), w1.make_dataset(val_ds, 1))
    d2 = (w2.make_dataset(test_ds, 1), w2.make_dataset(train_ds, batch_size), w2.make_dataset(val_ds, 1))
    return (d1, d2)