import math

import numpy as np
import tensorflow as tf

SEED = 42

# Adapted from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               columns, normalise=True,
               feature_columns=None, label_columns=None):

        # Work out the label column indices.
        self.label_columns = label_columns
        self.feature_columns = feature_columns
        self.column_indices = {name: i for i, name in
                            enumerate(columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.normalise = normalise

        self.total_window_size = input_width + shift + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        if self.feature_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.feature_columns],
                axis=-1
            )
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def window_dataset(self, segments) -> tf.data.Dataset:
        tf_ds = None
        for s in segments:
            t_tf_ds: tf.data.Dataset = tf.keras.utils.timeseries_dataset_from_array(
                np.array(s, dtype=np.float32), None,
                sequence_length=self.total_window_size, sequence_stride=1
            ).unbatch()
            if tf_ds is None:
                tf_ds = t_tf_ds
            else:
                tf_ds = tf_ds.concatenate(t_tf_ds)
        return tf_ds

    def make_dataset(self, data, batch_size=8, shuffle=True):
        tf_ds = None
        length = 0
        for _, v in data.items():
            length = max(length, len(v))
            if tf_ds is None:
                tf_ds = self.window_dataset(v)
            else:
                tf_ds = tf_ds.concatenate(self.window_dataset(v))

        assert tf_ds is not None
        if shuffle:
            tf_ds = tf_ds.shuffle(math.ceil(length / 2))
        tf_ds = tf_ds.batch(batch_size).map(self.split_window)

        return tf_ds

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])