PASCIFAR
========

Extract the PASCAL VOC 2012 compatible classes from CIFAR 10 and CIFAR 100 train datasets.

Creates a new dataset called PASCIFAR (PASCAL + CIFAR) that can be used to test classification algorithms trained on the PASCAL VOC 2012 dataset.

There are 3 missing categories from the PASCAL VOC 2012 dataset:

1. cow
2. pottedplant
3. sheep

PASCIFAR, thus, covers 17/20 PASCAL VOC 2012 classes.

# Relations

PASCAL VOC 2012 | CIFAR-10 | CIFAR-100
--- | --- | ---
airplane  | airplane | -
bicycle  | - | bicycle
bird  | bird | -
boat  | ship | -
bottle | - | food containers (bottles)
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
sofa | - | -
train | - | vehicles 1 (train)
tvmonitor | - | household electrical devices (television)

# Usage

```python
python builder.sh
```

It downloads the required CIFAR datasets and build PASCIFAR in the current directory.

# Copyright

CIFAR-10 and CIFAR-100 images has been collected by Alex Krizhevsky: https://www.cs.toronto.edu/~kriz/cifar.html

PASCIFAR `builder.py` is released under Mozilla Public License 2.0.
