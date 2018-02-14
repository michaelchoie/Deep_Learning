"""Import color images of house numbers collected from Google street view."""

import os
from os.path import isdir, isfile
from tqdm import tqdm
from urllib.request import urlretrieve


class DLProgress(tqdm):
    """To create download progress bar."""

    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        To create download progress bar.

        In urlretrieve, the hook is passed three arguments:
            1) Count of blocks transferred so far
            2) Block size in bytes
            3) Total size of the file
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_data(data_dir, url_train, url_test):
    """Download SVHN dataset."""
    if not isfile(data_dir + "train_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1,
                        desc='SVHN Training Set') as pbar:
            urlretrieve(url_train, data_dir + "train_32x32.mat", pbar.hook)

    if not isfile(data_dir + "test32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1,
                        desc='SVHN Testing Set') as pbar:
            urlretrieve(url_test, data_dir + "test_32x32.mat", pbar.hook)


def main():
    """Download and save data."""
    data_dir = "data/"
    url_train = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    url_test = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

    if not isdir(data_dir):
        os.makedirs("./data")

    download_data(data_dir, url_train, url_test)

    print("Download complete.")

if __name__ == "__main__":
    main()
