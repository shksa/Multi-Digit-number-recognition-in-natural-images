import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.models import Model

# pickle_file = 'SVHN_64x64x1_test.pickle'

# with open(pickle_file, 'rb') as f:
#     save = pickle.load(f)
#     test_dataset = save['train_dataset']
#     test_labels = save['train_labels']
#     del save  # hint to help gc free up memory
#     print('Test set', test_dataset.shape, test_labels.shape)

pickle_file = 'SVHN_64x64x1_valid.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    test_dataset = save['train_dataset']
    test_labels = save['train_labels']
    del save  # hint to help gc free up memory
    print('Test set', test_dataset.shape, test_labels.shape)

test_dataset = test_dataset.astype('float32')
test_labels = test_labels.astype('uint8')

image_width = 64
image_height = 64
num_labels = 11 # 0-9, + blank 
num_channels = 1 # greyscale

depth1 = 32
depth2 = 48
depth3 = 64
depth4 = 80
depth5 = 128
depth6 = 144
depth7 = 160

he = 'he_normal'
w = 3

x = Input(shape = (image_width, image_height, num_channels))

y = Convolution2D(filters=depth1, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block1_conv1")(x)
y = Activation('relu', name="block1_act1")(y)
y = BatchNormalization(name="block1_bnorm1")(y)
y = MaxPooling2D(name="block1_pool1")(y)
y = Dropout(0.2, name="block1_drop1")(y)

y = Convolution2D(filters=depth2, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block2_conv1")(y)
y = Activation('relu', name="block2_act1")(y)
y = BatchNormalization(name="block2_bnorm1")(y)
y = Convolution2D(filters=depth3, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block2_conv2")(y)
y = Activation('relu', name="block2_act2")(y)
y = BatchNormalization(name="block2_bnorm2")(y)
y = MaxPooling2D(name="block2_pool1")(y)
y = Dropout(0.4, name="block2_drop2")(y)

y = Convolution2D(filters=depth4, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block3_conv1")(y)
y = Activation('relu', name="block3_act1")(y)
y = BatchNormalization(name="block3_bnorm1")(y)
y = MaxPooling2D(name="block3_pool1")(y)
y = Dropout(0.3, name="block3_drop1")(y)
y = Convolution2D(filters=depth5, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block3_conv2")(y)
y = Activation('relu', name="block3_act2")(y)
y = BatchNormalization(name="block3_bnorm2")(y)
y = MaxPooling2D(name="block3_pool2")(y)
y = Dropout(0.4, name="block3_drop2")(y)

y = Convolution2D(filters=depth6, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block4_conv1")(y)
y = Activation('relu', name="block4_act1")(y)
y = BatchNormalization(name="block4_bnorm1")(y)
y = Dropout(0.3, name="block4_drop1")(y)
y = Convolution2D(filters=depth7, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block4_conv2")(y)
y = Activation('relu', name="block4_act2")(y)
y = BatchNormalization(name="block4_bnorm2")(y)
y = MaxPooling2D(name="block4_pool1")(y)
y = Dropout(0.5, name="block4_drop2")(y)

h = Flatten(name="feature_vector")(y)

probs1 = Dense(units = num_labels, kernel_initializer = he, activation="softmax", name="digit1")(h)
probs2 = Dense(units = num_labels, kernel_initializer = he, activation="softmax", name="digit2")(h)
probs3 = Dense(units = num_labels, kernel_initializer = he, activation="softmax", name="digit3")(h)
probs4 = Dense(units = num_labels, kernel_initializer = he, activation="softmax", name="digit4")(h)
probs5 = Dense(units = num_labels, kernel_initializer = he, activation="softmax", name="digit5")(h)

out = [probs1, probs2, probs3, probs4, probs5]
model = Model(inputs=x, outputs=out)

# model.summary()
model.load_weights("saved/phase_4.hdf5")
print('initialized')

batch_size = 512

num_batches = np.floor(test_dataset.shape[0] / batch_size).astype('uint8')
print(num_batches)

j = 0
count = 0
count_ = 0
tempc_ = 0
tempc__ = 0
for i in range(num_batches):
	batch_predictions = model.predict_on_batch(x = test_dataset[j:j+batch_size])
	preds_array = np.asarray(batch_predictions)
	predictions = np.argmax(preds_array, 2).T
	labels = test_labels[j:j+batch_size, 1:6]
	temp_c = 0
	temp_c_ = 0
	for k in range(batch_size):
		if np.array_equal(predictions[k], labels[k]) :
			count += 1
			temp_c += 1
		else:
			count_ += 1
			temp_c_ += 1
	tempc_ += temp_c
	tempc__ += temp_c_
	print(batch_size, temp_c, temp_c_)
	print(accuracy(preds_array, test_labels[j:j+batch_size, 1:6]))
	j+=batch_size
print(test_labels.shape[0])
print(count)
print(count_)
print(tempc_)
print(tempc__)


