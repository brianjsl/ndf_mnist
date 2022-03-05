# Image Correspondance in Overlapping MNIST with NDFs

This is the code for finding image correspondence on an overlapping MNIST dataset.

## Generating Data
Run generator.py as follows:
```
python3 generator.py --overlapmnist_path ./data/MNIST/overlapMNIST --num_image_per_class 1000 --image_size 36 36 
```
This will create a dataset of 36x36 images of overlapping mnist digits in folders of the form 'ab' where 'a' is the
left digit and 'b' is the right digit. 

The corresponding dataloader class can be found in dataloader.py which will turn the labels into labels usable in 
multilabel classification (in the ResNet block). 

## Miscallaneous Issues:

### To fix g++ issues on satori
#### if you get some c++ compiler warning
export CXX=g++

#### cuda home not set
export CUDA_HOME=/software/cuda/11.4

#### libcudart.so.11.0 issues
export LD_LIBRARY_PATH=/software/cuda/11.4/targets/ppc64le-linux/lib/