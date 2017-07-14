import pickle
import gzip
import array

# This script dumps the mnist database into flat data + labels files on disk.
# The format of the output is a flat array of 28x28 floats, and a list of int8.
# The mnist database can be found here:
# http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


def dumpToFile(dataset):
    data, labels = dataset

    imagesFile = open('mnist_images.bin', 'wb')
    data.tofile(imagesFile)
    imagesFile.close()

    labelsFile = open('mnist_labels.bin', 'wb')
    L = array.array('B', labels)
    L.tofile(labelsFile)
    labelsFile.close()

with gzip.open("./mnist.pkl.gz", 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    dumpToFile(train_set)

