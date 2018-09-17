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
from keras.utils import Sequence
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

model = keras.models.Sequential([

		Conv2D(8, (3, 3), padding = "same", input_shape = input_shape),
		Activation("relu"),
		BatchNormalization(),
		MaxPooling2D(pool_size = (2, 2)),
		Dropout(0.25),

		Conv2D(16, (3, 3), padding = "same"),
		Activation("relu"),
		BatchNormalization(),

		Conv2D(16, (3, 3), padding = "same"),
		Activation("relu"),
		BatchNormalization(),
		MaxPooling2D(pool_size = (2, 2)),
		Dropout(0.25),

		Flatten(),
		Dense(128),
		Activation("relu"),
		BatchNormalization(),
		Dropout(0.5),
		Dense(64),
		Activation("relu"),
		BatchNormalization(),
		Dropout(0.5),
		Dense(1),
		Activation("sigmoid")

	])

"""

	Training

"""


epochs = 50
learning_rate = 1e-3
batch_size = 32

class DataGenerator(Sequence):
	def __init__(self, path_names):
		self.path_names = path_names
		self.indexes = np.arange(len(self.path_names))
		np.random.shuffle(self.indexes)

	def __len__(self):
		return int(np.floor(len(self.path_names) / batch_size))

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.path_names))
		np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		batch_indexes = self.indexes[index * batch_size: (index + 1) * batch_size]
		X = np.empty((batch_size, image_utils.img_height, image_utils.img_width, image_utils.img_depth), dtype="float32")
		y = np.empty((batch_size), dtype="uint8")

		for i, batch_i in enumerate(batch_indexes):
			X[i,] = np.load(os.path.join("data", self.path_names[batch_i]))
			y[i] = 1 if self.path_names[0] == 'c' else 0

		return X, y



paths = os.listdir("data")

first_car = paths.index('c0')
non_car_paths = paths[:first_car]
car_paths = paths[first_car:]

np.random.shuffle(car_paths)
np.random.shuffle(non_car_paths)
split_point_non_car = int(len(non_car_paths) * 0.8)
split_point_car = int(len(car_paths) * 0.8)

training_generator = DataGenerator(non_car_paths[:split_point_non_car] + car_paths[:split_point_car])
testing_generator = DataGenerator(non_car_paths[split_point_non_car:] + car_paths[split_point_car:])

optimizer = Adam(lr=learning_rate, decay=learning_rate/epochs)

# Init model
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

# Train

H = model.fit_generator(
	generator=training_generator,
	validation_data=testing_generator,
	steps_per_epoch=(split_point_non_car + split_point_car) / batch_size,
	epochs=epochs, verbose=1)

model.save("model")
