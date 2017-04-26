import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.models import Model

pickle_file = 'SVHN_64x64x1_train.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    del save  # hint to help gc free up memory
    print('Test set', train_dataset.shape, train_labels.shape)

print(np.max(train_dataset[0]))

train_dataset = train_dataset.astype('float32')
train_labels = train_labels.astype('uint8')

y0 = np.reshape(train_labels[:, 1], (97722, 1))
y1 = np.reshape(train_labels[:, 2], (97722, 1))
y2 = np.reshape(train_labels[:, 3], (97722, 1))
y3 = np.reshape(train_labels[:, 4], (97722, 1))
y4 = np.reshape(train_labels[:, 5], (97722, 1))

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
w = 4

x = Input(shape = (image_width, image_height, num_channels))

y = Convolution2D(filters=depth1, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block1_conv1")(x)
y = Activation('relu', name="block1_act1")(y)
y = BatchNormalization(name="block1_bnorm1")(y)
y = MaxPooling2D(name="block1_pool1")(y)
y = Dropout(0.2, name="block1_drop1")(y)

y = Convolution2D(filters=depth2, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block2_conv1")(y)
y = Activation('relu', name="block2_act1")(y)
y = BatchNormalization(name="block2_bnorm1")(y)
y = Dropout(0.25, name="block2_drop1")(y)
y = Convolution2D(filters=depth3, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block2_conv2")(y)
y = Activation('relu', name="block2_act2")(y)
y = BatchNormalization(name="block2_bnorm2")(y)
y = MaxPooling2D(name="block2_pool1")(y)
y = Dropout(0.25, name="block2_drop2")(y)

y = Convolution2D(filters=depth4, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block3_conv1")(y)
y = Activation('relu', name="block3_act1")(y)
y = BatchNormalization(name="block3_bnorm1")(y)
y = MaxPooling2D(name="block3_pool1")(y)
y = Dropout(0.25, name="block3_drop1")(y)
y = Convolution2D(filters=depth5, kernel_size=(3, 3), kernel_initializer=he, kernel_constraint=maxnorm(w), padding='same', name="block3_conv2")(y)
y = Activation('relu', name="block3_act2")(y)
y = BatchNormalization(name="block3_bnorm2")(y)
y = MaxPooling2D(name="block3_pool2")(y)
y = Dropout(0.25, name="block3_drop2")(y)

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

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

checkpointer = ModelCheckpoint(filepath="saved/phase_5.hdf5", monitor='loss', verbose=1, save_best_only=True)

K_train_labels = [y0, y1, y2, y3, y4]

training_stats = model.fit(x=train_dataset, y=K_train_labels, epochs=1, batch_size=64, verbose=1, callbacks=[checkpointer])

#model.save_weights('SVHN_model_weights.h5')
#json_string = model.to_json()
# text_file = open("SVHN_model_json", "w")
# text_file.write(json_string)
# text_file.close()




