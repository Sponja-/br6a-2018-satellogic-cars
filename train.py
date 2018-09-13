import keras
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import image_utils
import random
import os

input_shape = (image_utils.img_width, image_utils.img_height, image_utils.img_depth)


"""

	Modelo de la red

"""

model = keras.models.Sequential()

# Bloque de CONV => RELU => POOL
model.add(Conv2D(8, (3, 3), padding = "same", input_shape = input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Bloque CONV => RELU
model.add(Conv2D(16, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization())

# Bloque CONV => RELU => POOL
model.add(Conv2D(16, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization())

# POOL

model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Pasa a capas FC

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))


"""

	Training

"""

epochs = 100
learning_rate = 1e-3
batch_size = 32

data = []
car_paths = os.listdir("true_segments")
non_car_paths = os.listdir("false_segments")

for path in car_paths:
	data.append(np.load(os.path.join("true_segments", path)))

for path in non_car_paths:
	data.append(np.load(os.path.join("false_segments", path)))

labels = [[1]] * len(car_paths) + [[0]] * len(non_car_paths)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
train_x, train_y, test_x, test_y = train_test_split(data, labels, test_size=0.2)

augmentator = ImageDataGenerator(rotation_range=25, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

optimizer = Adam(lr=learning_rate, decay=learning_rate/epochs)

# Init model
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train

H = model.fit_generator(
	augmentator.flow(train_x, train_y, batch_size=batch_size),
	validation_data=(test_x, test_y),
	steps_per_epoch=len(train_x) / batch_size,
	epochs=epochs, verbose=1)

model.save("model")
