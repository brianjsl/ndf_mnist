import subprocess
import os
import os.path as osp
import numpy as np
# from imageio import imwrite
import argparse

parser = argparse.ArgumentParser()

mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 
            't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

