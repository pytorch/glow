import pickle
import gzip
import array
import urllib
import tarfile
import os.path

# This script downloads and extracts the mnist, ptb and cifar-10 databases.

print("""Downloading test files. If the download fails try setting up a proxy:
    #export http_proxy="http://fwdproxy:8080

""")

mnist_filename = "mnist.pkl.gz"
cifar10_filename = "cifar-10.binary.tar.gz"
ptb_filename = "ptb.tgz"

if os.path.exists(mnist_filename):
    print("MNIST file found. Not downloading.")
else:
    print("Downloading MNIST ... ")
    urllib.urlretrieve ("http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz", mnist_filename)

if os.path.exists(cifar10_filename):
    print("CIFAR file found. Not downloading.")
else:
    print("Downloading CIFAR ... ")
    urllib.urlretrieve ("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", cifar10_filename)

if os.path.exists(ptb_filename):
    print("PTB file found. Not downloading.")
else:
    print("Downloading PTB ... ")
    urllib.urlretrieve ("http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz", ptb_filename)

def dumpToFile(dataset):
    data, labels = dataset

    imagesFile = open('mnist_images.bin', 'wb')
    data.tofile(imagesFile)
    imagesFile.close()

    labelsFile = open('mnist_labels.bin', 'wb')
    L = array.array('B', labels)
    L.tofile(labelsFile)
    labelsFile.close()

print("Extracting the mnist database.")

with gzip.open(mnist_filename, 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    dumpToFile(train_set)


print("Extracting the CIFAR-10 database.")
tar = tarfile.open(cifar10_filename, "r:gz")
tar.extractall()
tar.close()


print("Extracting the PTB database.")
tar = tarfile.open(ptb_filename, "r:gz")
tar.extractall('ptb')
tar.close()
