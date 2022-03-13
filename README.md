# Image Correspondance in Overlapping MNIST with NDFs

This is the code for finding image correspondence on an overlapping MNIST dataset.

## Generating Data
Run generator2.py as follows:
```
python3 generator2.py --overlapmnist_path ./data/MNIST/overlapMNIST --num_image_per_class 1000 --image_size 32 32 
```
This will create a dataset of 32x32 images of overlapping mnist digits in folders of the form 'ab' where 'a' is the
left digit and 'b' is the right digit. 

The corresponding dataloader class can be found in dataloader.py which will turn the labels into labels usable in 
multilabel classification (in the ResNet block). 

## Training the Neural Field
Run neural_field.py for around 20 epochs. The resulting checkpoints will be saved under checkpoints/chkpt_{}.pt. 

## Reconstructing The Images
Run reconstruction.py as follows:
```
python3 reconstruction.py --imagenum n
```
where n is the image number you want to reproduce in dataloader_ndf.py. 

## Doing the Energy Optimization
Run optimizer.py as follows:
```
python3 optimizer.py --image1num n1 --image2num n2
```
with image numbers n1 and n2 of your choice. You will be prompted to a screen where you can click and select points.
Then, an energy optimization will be run on the second image and the corresponding points will be chosen and an image 
with those points labeled will be stored in ./src/.

## Miscallaneous Issues:

### To fix g++ issues on satori
#### if you get some c++ compiler warning
```
export CXX=g++
```
#### cuda home not set
```
export CUDA_HOME=/software/cuda/11.4
```
#### libcudart.so.11.0 issues
```
export LD_LIBRARY_PATH=/software/cuda/11.4/targets/ppc64le-linux/lib/
```