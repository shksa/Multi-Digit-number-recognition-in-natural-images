## actual321.ipynb has the training for the model on images with 1, 2, 3 numbers in it. The training routine is different from the usual method. A digit classifier gets backprop'ed only if it's the right classifier for the digit.
## ex: for 2 digit image, only the first 2 digit classifier's are backprop'ed. 3rd, 4th, 5th are not backprop'ed.
# saliency map.ipynb has the saliency visualization for the model trained in actual321.ipynb.
## See SVHN folder for experiments on different models for the same task.(More interesing!).

## model.py in this folder is a straight forward approach that achieves very high accuracy
## Multi-Digit-number-recognition-in-natural-images

## Acheived a test accuracy of 91%
## Run test.py to check accuray
## Test.ipynb is nice visualization of the model accuracy
## The trained weights are in the file "phase_4.hdf5"

#### Used batch-normalization and dropout
#### Used max-norm constraint on weights instead of L2 regularization
#### He-normal initialization
#### 5 classifiers for 5 digits
#### Absence of a digit is realized by using digit "10"
