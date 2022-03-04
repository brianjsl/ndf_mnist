import subprocess
import os
import numpy as np
# from imageio import imwrite
import argparse

mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 
            't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

def check_mnist_dir(data_dir):
    '''
    Check if MNIST is downloaded.
    '''
    downloaded = np.all([os.path.isfile(os.path.join(data_dir, key)) for key in mnist_keys])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST was found')

def download_mnist(data_dir):
    '''
    Downloads the MNIST dataset into the data directory.
    '''
    data_url = 'http://yann.lecun.com/exdb/mnist/'
    for k in mnist_keys:
        k += '.gz'
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gunzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

def argparser():   
    '''
    Argparser Setup. 
    
    Arguments: 
        --mnist_path: path to MNIST dataset
        --overlapmnist_path: path to store overlapping MNIST dataset
        --train_val_test_ratio: ratio (in percentage) of train to val to test
        --image_size: default image size
        --num_image_per_class: number of images per class
        --random_seed: default random seed to choose    
    '''

    #initialize argparser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    

    parser.add_argument('--mnist_path', type = str, 
                        default = './data/MNIST/raw',
                        help='path to *.gz files'
                        )
    parser.add_argument('--overlapmnist_path', type = str,
                        default = './data/MNIST/overlapMNIST',
                        help = 'path to overlapping files'
                        )
    parser.add_argument('--train_val_test_ratio', type=int, nargs='+',
                        default=[64, 16, 20], help='percentage')
    parser.add_argument('--image_size', type=int, nargs='+',
                        default=[28, 28])
    parser.add_argument('--num_image_per_class', type=int, default=10000)
    parser.add_argument('--random_seed', type=int, default=123)


    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = argparser()
