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
Run neural_field.py for around 40 epochs. The resulting checkpoints will be saved under checkpoints/chkpt_{}.pt. 
UPDATE: You can download a pre-trained model [here](https://www.dropbox.com/s/cof2ctfwdesmzix/chkpt_39.pt?dl=0). 
Place the downloaded checkpoint in the 'checkpoints' folder.

## Reconstructing The Images
Run reconstruction.py as follows:
```
python3 reconstruction.py --imagenum n
```
where n is the image number you want to reproduce in dataloader_ndf.py. 

## Doing the Energy Optimization
Run optimizer.py as follows:
```
python3 optimizer.py --image1_class c1 --image1_num n1 --image2_class c2 --image2_num n2
```
with image numbers n1 and n2 of your choice. For example, for image1 to be of class 07, image number 37 and
image2 to have class 31, we can run the following:
```
python3 optimizer.py --image1_class 07 --image1_num 79 --image2_class 37 --image2_num 31
```
You will be prompted to a screen where you can click and select points. Then, an energy optimization will be run on the second image and the corresponding points will be chosen and an image with those points will appear for side by side comparison.

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