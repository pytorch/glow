#!/usr/bin/env python
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

from __future__ import division, print_function

import argparse
import array
import collections
import gzip
import os.path
import pickle
import sys
import tarfile


try:
    from urllib.error import URLError
except ImportError:
    from urllib2 import URLError

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

Dataset = collections.namedtuple("TargetItem", "filename, url, handler, dest_path")


# Load a file using pickle module, and parameters vary based on different
# Python versions.
def pickle_load(file):
    if sys.version_info.major >= 3:
        return pickle.load(file, encoding="bytes")
    return pickle.load(file)


# A helper function to extract mnist dataset from tar file, and split the dataset
# into data and labels.
def handle_mnist(filename, dest_path):
    print("Extracting {} ...".format(filename))
    with gzip.open(filename, "rb") as file:
        training_set, _, _ = pickle_load(file)
        data, labels = training_set

        images_file = open(os.path.join(dest_path, "mnist_images.bin"), "wb")
        data.tofile(images_file)
        images_file.close()

        labels_file = open(os.path.join(dest_path, "mnist_labels.bin"), "wb")
        L = array.array("B", labels)
        L.tofile(labels_file)
        labels_file.close()


def untar(filename, dest_path, member=None):
    print("Extracting {} ...".format(filename))
    tar = tarfile.open(filename, "r:gz")
    if not member:
        tar.extractall(dest_path)
    else:
        tar.extract(member, dest_path)
    tar.close()


DATASETS = dict(
    mnist=Dataset(
        "mnist.pkl.gz",
        "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz",
        handle_mnist,
        ".",
    ),
    cifar10=Dataset(
        "cifar-10.binary.tar.gz",
        "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        untar,
        ".",
    ),
    ptb=Dataset(
        "ptb.tgz",
        "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz",
        untar,
        "ptb",
    ),
    fr2en=Dataset(
        "fr2en.tar.gz",
        "http://fb-glow-assets.s3.amazonaws.com/models/fr2en.tar.gz",
        untar,
        "fr2en",
    ),
)


DATASET_NAMES = list(DATASETS.keys())
CAFFE2_MODELS = [
    "densenet121",
    "inception_v1",
    "inception_v2",
    "lenet_mnist",
    "resnet50",
    "shufflenet",
    "squeezenet",
    "vgg19",
    "zfnet512",
    "bvlc_alexnet",
    "en2gr",
    "quant_resnet50",
]
ONNX_MODELS = [
    "resnet50",
    "vgg19",
    "squeezenet",
    "zfnet512",
    "densenet121",
    "shufflenet",
    "inception_v1",
    "inception_v2",
    "bvlc_alexnet",
    "lenet_mnist",
    "googlenet_v1_slim",
    "googlenet_v4_slim",
    "resnet50_tf",
    "emotion_ferplus",
    "bvlc_reference_rcnn_ilsvrc13",
]


def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download(path, filename, url):
    if not os.path.exists(path):
        os.mkdir(path)
    destFile = os.path.join(path, filename)
    if os.path.exists(destFile):
        print("{} already exists, skipping ...".format(filename))
    else:
        print("Downloading {} from {} ...".format(filename, url))
        try:
            urlretrieve(url, destFile, reporthook=report_download_progress)
        except URLError:
            print("Error downloading {}!".format(filename))
        finally:
            # Just a newline.
            print()


def download_caffe2_models(outDir, models):
    for modelname in models:
        print("For model ", modelname);
        for filename in ["predict_net.pbtxt", "predict_net.pb", "init_net.pb"]:
            path = os.path.join(outDir, modelname)
            url = "http://fb-glow-assets.s3.amazonaws.com/models/{}/{}".format(
                modelname, filename
            )
            download(path, filename, url)
        if modelname == "en2gr":
            for filename in ["dst_dictionary.txt", "src_dictionary.txt"]:
                path = os.path.join(outDir, "en2gr")
                url = "http://fb-glow-assets.s3.amazonaws.com/models/en2gr/{}".format(
                    filename
                )
                download(path, filename, url)
    return


def download_onnx_models(outDir, models):
    for modelname in models:
        if modelname in [
            "resnet50",
            "vgg19",
            "squeezenet",
            "zfnet512",
            "densenet121",
            "shufflenet",
        ]:
            url = "https://s3.amazonaws.com/download.onnx/models/opset_6/{}.tar.gz".format(
                modelname
            )
            filename = "{}.tar.gz".format(modelname)
            download(outDir, filename, url)
            untar(os.path.join(outDir, filename), outDir)
        elif modelname in ["inception_v1", "inception_v2", "bvlc_alexnet"]:
            url = "https://s3.amazonaws.com/download.onnx/models/opset_8/{}.tar.gz".format(
                modelname
            )
            filename = "{}.tar.gz".format(modelname)
            download(outDir, filename, url)
            untar(os.path.join(outDir, filename), outDir)
        elif modelname in ["lenet_mnist"]:
            url = "http://fb-glow-assets.s3.amazonaws.com/models/{}.tar.gz".format(
                modelname
            )
            filename = "{}.tar.gz".format(modelname)
            download(outDir, filename, url)
            untar(os.path.join(outDir, filename), outDir)

        elif modelname in ["googlenet_v1_slim", "googlenet_v4_slim", "resnet50_tf"]:
            url = "http://fb-glow-assets.s3.amazonaws.com/models/{}.onnx".format(
                modelname
            )
            filename = "{}.onnx".format(modelname)
            path = os.path.join(outDir, modelname)
            download(path, filename, url)
        elif modelname == "emotion_ferplus":
            url = "https://onnxzoo.blob.core.windows.net/models/opset_8/emotion_ferplus/emotion_ferplus.tar.gz"
            filename = "emotion_ferplus.tar.gz"
            download(outDir, filename, url)
            untar(os.path.join(outDir, filename), outDir, "emotion_ferplus/model.onnx")
        elif modelname == "bvlc_reference_rcnn_ilsvrc13":
            url = "https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz"
            filename = "bvlc_reference_rcnn_ilsvrc13.tar.gz"
            download(outDir, filename, url)
            untar(
                os.path.join(outDir, filename),
                outDir,
                "bvlc_reference_rcnn_ilsvrc13/model.onnx",
            )
    return


def parse():
    parser = argparse.ArgumentParser(description="Download datasets for Glow")
    parser.add_argument("-d", "--datasets", nargs="+", choices=DATASET_NAMES)
    parser.add_argument("-D", "--all-datasets", action="store_true")
    parser.add_argument("-c", "--caffe2-models", nargs="+", choices=CAFFE2_MODELS)
    parser.add_argument("-C", "--all-caffe2-models", action="store_true")
    parser.add_argument("-o", "--onnx-models", nargs="+", choices=ONNX_MODELS)
    parser.add_argument("-O", "--all-onnx-models", action="store_true")
    parser.add_argument("-P", "--output-directory", default=".")
    options = parser.parse_args()
    if options.all_datasets:
        datasets = DATASET_NAMES
    elif options.datasets:
        datasets = options.datasets
    else:
        datasets = []

    if options.all_caffe2_models:
        caffe2Models = CAFFE2_MODELS
    elif options.caffe2_models:
        caffe2Models = options.caffe2_models
    else:
        caffe2Models = []

    if options.all_onnx_models:
        onnxModels = ONNX_MODELS
    elif options.onnx_models:
        onnxModels = options.onnx_models
    else:
        onnxModels = []

    return options.output_directory, datasets, caffe2Models, onnxModels


def main():
    outDir, datasets, caffe2Models, onnxModels = parse()
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    outDir = os.path.join(".", outDir)
    try:
        for name in datasets:
            dataset = DATASETS[name]
            download(outDir, dataset.filename, dataset.url)
            dataset.handler(
                os.path.join(outDir, dataset.filename),
                os.path.join(outDir, dataset.dest_path),
            )
        if datasets:
            print("\n===Done with downloading datasets.\n\n")
        if caffe2Models:
            download_caffe2_models(outDir, caffe2Models)
            print("===Done with downloading caffe2 models.\n\n")
        if onnxModels:
            download_onnx_models(outDir, onnxModels)
            print("===Done with downloading onnx models.\n\n")
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
