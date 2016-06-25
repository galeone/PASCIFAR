PASCIFAR
========

Extract the PASCAL VOC 2012 compatible classes from CIFAR 10 and CIFAR 100 train datasets.

Creates a new dataset called PASCIFAR (PASCAL + CIFAR) that can be used to test classification algorithms trained on the PASCAL VOC 2012 dataset.

There are 4 missing categories from the PASCAL VOC 2012 dataset:

1. cow
2. pottedplant
3. sheep
4. boat

PASCIFAR, thus, covers 16/20 PASCAL VOC 2012 classes.

The `ts.csv` file contains the `file/path.png, label_id` where `label_id` has the value that's the position of the label in the PASCAL VOC dataset sorted list of labels.

Using the following format you can avoid to convert labels between datasets.

# Download

A PASCIFAR archive is avaiable here: http://download.nerdz.eu/PASCIFAR.tgz

# Relations

PASCAL VOC 2012 | CIFAR-10 | CIFAR-100
--- | --- | ---
airplane  | airplane | -
bicycle  | - | bicycle
bird  | bird | -
boat  | - | -
bottle | - | food containers (bottle)
bus | - | vehicles 1 (bus)
car | car | -
cat | cat | -
chair | - | household furniture (chair)
cow | - | -
diningtable | - | household furniture (table)
dog | dog | -
horse | horse | -
motorbike | - | vehicles 1 (motorcycle)
person | - | people (macroclass)
pottedplan | - | -
sheep | - | -
sofa | - | houseold furniture (couch)
train | - | vehicles 1 (train)
tvmonitor | - | household electrical devices (television)

# Usage

```python
python builder.sh
```

It downloads the required CIFAR datasets and build PASCIFAR in the current directory.

# Dataset structure and content

PASCIFAR uses the simplest structure: 1 folder for each class.

Every image is in `.png` format and have the fixed size of the CIFAR dataset: `32x32x3`.
Each folder has the name of the corresponding PASCAL VOC 2012 class.

Here's the detailed content:

Class | #
--- | ---
car | 5000
cat | 5000
bird | 5000
sofa | 500
person | 2500
bottle | 500
train | 500
horse | 5000
motorbike | 500
chair | 500
bus | 500
dog | 5000
bicycle | 500
tvmonitor | 500
diningtable | 500
aeroplane | 5000
Total | 42000

# Copyright

CIFAR-10 and CIFAR-100 images has been collected by Alex Krizhevsky: https://www.cs.toronto.edu/~kriz/cifar.html

PASCIFAR `builder.py` is released under Mozilla Public License 2.0.
