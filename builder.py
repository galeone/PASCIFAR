#PASCIFAR: a subset of CIFAR-10 and CIFAR-100 datasets, PASCAL VOC
#2012 compatible.
#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Build the PASCIFAR dataset"""

import csv
import sys
import os
import tarfile
import pickle
import numpy as np
from PIL import Image
from six.moves import urllib

# pascal sorted labels
PASCAL_LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                 "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                 "motorbike", "person", "pottedplant", "sheep", "sofa",
                 "train", "tvmonitor"]

# use automobile to replace car
CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "dog", "horse"]

# use couch instead of sofa
CIFAR100_FINE_LABELS = ["bicycle", "bottle", "bus", "chair", "table",
                        "motorcycle", "couch", "train", "television"]
# use people to replace person
CIFAR100_COARSE_LABELS = ["people"]

# CIFAR2PASCAL label conversion using a dictionary:
CIFAR2PASCAL = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "bottle": "bottle",
    "bus": "bus",
    "automobile": "car",
    "cat": "cat",
    "chair": "chair",
    "table": "diningtable",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "people": "person",
    "couch": "sofa",
    "train": "train",
    "television": "tvmonitor",
}

PASCAL2PASCIFAR = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "MISSING",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "MISSING",
    "diningtable": "table",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "person": "people",
    "pottedplant": "MISSING",
    "sheep": "MISSING",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "television",
}


def get_labels():
    """Returns the CIFAR labels in a list"""
    labels = [key for key in CIFAR2PASCAL.keys()]
    labels.sort()
    return labels


# Adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10.py
def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = os.path.abspath(os.getcwd())
    if not os.path.exists(dest_directory +
                          "/cifar-10-batches-py/") and not os.path.exists(
                              dest_directory + "/cifar-100-python/"):
        files = ["cifar-100-python.tar.gz", "cifar-10-python.tar.gz"]
        for filename in files:
            filepath = os.path.join(dest_directory, filename)
            if not os.path.exists(filepath):

                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                        filename,
                        float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(
                "https://www.cs.toronto.edu/~kriz/" + filename, filepath,
                _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size,
                  'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print("Skipping download and extraction of CIFAR-{10,100}")
        print("cifar-10-batches-py and cifar-100-python already exist")


def unpickle(filename):
    """unpickle filename into a dictionary"""
    with open(filename, 'rb') as file_open:
        dictionary = pickle.load(file_open, encoding='latin1')
    return dictionary


def cifar2rgb(line):
    """Creates a valid RGB image from a line of 32*32*3 uint8"""
    return Image.fromarray(np.transpose(
        np.reshape(line, [32, 32, 3], order='F'), [1, 0, 2]))


def cifar10(dest, current_dir=os.path.abspath(os.getcwd())):
    """Extract from CIFAR-10 the elements with CIFAR10_LABELS and save it into dest.
    Inputs:
        dest: absolute path of PASCIFAR
        current_dir: directory where cifar-10-batches-py folder is
    Returns:
        None.
    """
    for label in CIFAR10_LABELS:
        os.mkdir(dest + "/" + CIFAR2PASCAL[label])

    counters = {label: 1 for label in CIFAR10_LABELS}

    labels = unpickle(current_dir + "/cifar-10-batches-py/batches.meta")[
        "label_names"]

    for i in range(1, 6):
        batch = unpickle(current_dir +
                         "/cifar-10-batches-py/data_batch_{}".format(i))
        # extract desired class only from the current batch
        for idx, line in enumerate(batch["data"]):
            label = labels[batch["labels"][idx]]
            if label in CIFAR10_LABELS:
                image = cifar2rgb(line)
                image.save(dest + "/" + CIFAR2PASCAL[label] + "/" + str(
                    counters[label]) + ".png")
                counters[label] += 1


def cifar100(dest, current_dir=os.path.abspath(os.getcwd())):
    """Extract from CIFAR-100 the elements with CIFAR100_LABELS and save it into dest
    Inputs:
        dest: absolute path of PASCIFAR
        current_dir: directory where cifar-100-python folder is
    Returns:
        None.
    """
    cifar100_labels = CIFAR100_FINE_LABELS + CIFAR100_COARSE_LABELS
    for label in cifar100_labels:
        os.mkdir(dest + "/" + CIFAR2PASCAL[label])

    counters = {label: 1 for label in cifar100_labels}

    meta_dict = unpickle(current_dir + "/cifar-100-python/meta")
    fine_labels = meta_dict["fine_label_names"]
    coarse_labels = meta_dict["coarse_label_names"]

    batch = unpickle(current_dir + "/cifar-100-python/train")
    # extract desired class only from the current batch
    for idx, line in enumerate(batch["data"]):
        fine_label = fine_labels[batch["fine_labels"][idx]]
        coarse_label = coarse_labels[batch["coarse_labels"][idx]]
        is_fine = fine_label in CIFAR100_FINE_LABELS
        is_coarse = coarse_label in CIFAR100_COARSE_LABELS
        if is_fine or is_coarse:
            image = cifar2rgb(line)
            label = fine_label if is_fine else coarse_label
            image.save(dest + "/" + CIFAR2PASCAL[label] + "/" + str(counters[
                label]) + ".png")
            counters[label] += 1


def write_csv(dest):
    """Writes the ts.csv file, reading dest content that
    must be the PASCIFAR folder.
    ts.csv is saved into dest."""

    with open(dest + "/ts.csv", "w") as csv_file:
        writer = csv.DictWriter(csv_file, ["file", "label"])
        writer.writeheader()

        for label in get_labels():
            # use pascal label
            # it creates holes between pascifar labels but
            # ensure compatibility between the two datasets when using
            # numeric labels
            pascal_label = CIFAR2PASCAL[label]
            label_id = PASCAL_LABELS.index(pascal_label)
            path = dest + "/" + pascal_label + "/"
            images_count = len(next(os.walk(path))[2]) + 1
            for image in range(1, images_count):
                relative_path = pascal_label + "/" + str(image) + ".png"
                writer.writerow({"file": relative_path, "label": label_id})


def main():
    """Build the PASCAL compatible dataset.
    The resulting dataset containes 16/20 PASCAL compatible classes.
    There are 3 missing classes: boat, cow, pottedplant, sheep"""
    current_dir = os.path.abspath(os.getcwd())
    dest = current_dir + "/PASCIFAR"

    if not os.path.exists(dest):
        os.mkdir(dest)
        maybe_download_and_extract()
        cifar10(dest)
        cifar100(dest)
        write_csv(dest)
    else:
        print("PASCIFAR already built. Exit")


if __name__ == '__main__':
    sys.exit(main())
