# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import argparse
import array
import collections
import gzip
import os.path
import pickle
import sys
import tarfile
import urllib
import sys

try:
    from urllib.error import URLError
except ImportError:
    from urllib2 import URLError

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

Dataset = collections.namedtuple('Dataset', 'filename, url, handler, dest_path')


def pickle_load(file):
    if sys.version_info.major >= 3:
        return pickle.load(file, encoding='bytes')
    return pickle.load(file)


def handle_mnist(filename, dest_path):
    print('Extracting {} ...'.format(filename))
    with gzip.open(filename, 'rb') as file:
        training_set, _, _ = pickle_load(file)
        data, labels = training_set

        images_file = open('mnist_images.bin', 'wb')
        data.tofile(images_file)
        images_file.close()

        labels_file = open('mnist_labels.bin', 'wb')
        L = array.array('B', labels)
        L.tofile(labels_file)
        labels_file.close()


def untar(filename, dest_path):
    print('Extracting {} ...'.format(filename))
    tar = tarfile.open(filename, "r:gz")
    tar.extractall(dest_path)
    tar.close()


DATASETS = dict(
    mnist=Dataset(
        'mnist.pkl.gz',
        'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz',
        handle_mnist,
        '.',
    ),
    cifar10=Dataset(
        'cifar-10.binary.tar.gz',
        'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
        untar,
        '.',
    ),
    ptb=Dataset(
        'ptb.tgz',
        'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz',
        untar,
        'ptb',
    ),
    fr2en=Dataset(
        'fr2en.tar.gz',
        'http://fb-glow-assets.s3.amazonaws.com/models/fr2en.tar.gz',
        untar,
        'fr2en',
    ),
)
DATASET_NAMES = list(DATASETS.keys())


def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))


def download_dataset(dataset):
    if os.path.exists(dataset.filename):
        print('{} already exists, skipping ...'.format(dataset.filename))
    else:
        print('Downloading {} from {} ...'.format(dataset.filename,
                                                  dataset.url))
        try:
            urlretrieve(
                dataset.url,
                dataset.filename,
                reporthook=report_download_progress)
        except URLError:
            print('Error downloading {}!'.format(dataset.filename))
        finally:
            # Just a newline.
            print()


def parse():
    parser = argparse.ArgumentParser(description='Download datasets for Glow')
    parser.add_argument('-d', '--datasets', nargs='+', choices=DATASET_NAMES)
    parser.add_argument('-a', '--all', action='store_true')
    options = parser.parse_args()

    if options.all:
        datasets = DATASET_NAMES
    elif options.datasets:
        datasets = options.datasets
    else:
        parser.error('Must specify at least one dataset or --all.')

    return datasets


def main():
    datasets = parse()
    try:
        for name in datasets:
            dataset = DATASETS[name]
            download_dataset(dataset)
            dataset.handler(dataset.filename, dataset.dest_path)
        print('Done.')
    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    main()
