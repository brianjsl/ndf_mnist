# Neural Descriptor Fields on Overlapping MNIST_784.

This is the code for finding image correspondence on an overlapping MNIST dataset.

## Generating Data
Run generator.py as follows:
```
python3 generator.py --overlapmnist_path ./data/MNIST/overlapMNIST --num_image_per_class 1000 --image_size 42 42 
```
This will create a dataset of 42x42 images of overlapping mnist digits in folders of the form 'ab' where 'a' is the
left digit and 'b' is the right digit. 

The corresponding dataloader class can be found in dataloader.py which will turn the labels into labels usable in 
multilabel classification (in the ResNet block). 

