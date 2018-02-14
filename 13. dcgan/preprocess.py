"""Preprocess data for use in DCGAN."""
import numpy as np


class Dataset(object):
    """To scale pixels, partition for train/test, and form mini-batches."""

    def __init__(self, train, test, val_frac=0.8, shuffle=False,
                 scale_func=None):
        """Initalize object."""
        self.train_x = train['X']
        self.train_y = train['y']

        split_idx = int(len(test['y']) * (1 - val_frac))
        self.test_x = test['X'][:, :, :, :split_idx]
        self.valid_x = test['X'][:, :, :, split_idx:]
        self.test_y = test['y'][:split_idx]
        self.valid_y = test['y'][split_idx:]

        # Reallocate the axes at the start "position"
        self.train_x = np.rollaxis(self.train_x, 3)
        self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x = np.rollaxis(self.test_x, 3)

        if scale_func is None:
            self.scaler = self.scale
        else:
            self.scaler = scale_func

        self.shuffle = shuffle

    def scale(self, x, feature_range=(-1, 1)):
        """Scale pixel range."""
        x = ((x - x.min()) / (255 - x.min()))
        min, max = feature_range
        x = x * (max - min) + min

        return x

    def batches(self, batch_size):
        """Form mini-batches."""
        if self.shuffle:
            idx = np.arange(len(self.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]

        n_batches = len(self.train_y) // batch_size
        for ii in range(0, n_batches, batch_size):
            x = self.train_x[ii:ii + batch_size]
            y = self.train_y[ii:ii + batch_size]

            yield self.scaler(x), y
