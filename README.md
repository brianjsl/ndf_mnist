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

## Training the Latent Network
Run train_encoder.py to fine-tune ResNet on the data to create Latent Encodings. The model will be called 
olmnist_resnet.pt. You can evat luate the test accuracy by running test_resnet.py.

## Creating the Latent Encodings
Run create_latents.py to do inference on the trained ResNet model stored in src/models. 

## Point Sampling

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