import os

IMG_DIR = '../../data/MNIST/overlapMNIST/'
TRAIN_DIR = IMG_DIR + '/train/'
VAL_DIR = IMG_DIR + '/val/'
TEST_DIR = IMG_DIR + '/test/'

TEST_NAMES = []
TRAIN_NAMES = []
VAL_NAMES = []

for root, subdirs, files in os.walk(TEST_DIR):
    for subdir in subdirs:
        TEST_NAMES.append(subdir)

for root, subdirs, files in os.walk(TRAIN_DIR):
    for subdir in subdirs:
        TRAIN_NAMES.append(subdir)

for root, subdirs, files in os.walk(VAL_DIR):
    for subdir in subdirs:
        VAL_NAMES.append(subdir)


