import numpy as np
from skimage import io
from segmentation import segment, padded_segments
from keras.models import load_model
from argparse import ArgumentParser
import os

threshold = 0.2

model = load_model(os.path.join("models", "model_1"))

def count_cars(image, **kwargs):
	segments = segment(image)
	padded, segment_val = padded_segments(image, segments, list(range(segments.max() + 1)), mask=kwargs.get("mask", None))
	predictions = model.predict(padded)

	count = 0
	result = image.copy()
	for val, pred in zip(segment_val, predictions):
		result[segments == val] = [255 * pred[1], 255 * pred[0], 0]
		count += pred[0] > threshold

	return (count, result)

if __name__ == "__main__":
	image_paths = list(filter(lambda s: s.endswith(".jpg"), os.listdir("images")))
	mask_paths = list(filter(lambda s: s.endswith(".bmp"), os.listdir("tiles")))

	get_number = lambda elem: int(elem[elem.index('-') + 1:elem.index('.')])
	image_paths.sort(key=get_number)
	mask_paths.sort(key=get_number)

	for image_path, mask_path in zip(image_paths, mask_paths):
		image, mask = io.imread(os.path.join("images", image_path)), io.imread(os.path.join("tiles", mask_path))
		car_count, car_image = count_cars(image, mask=mask)
		print(car_count, car_image)
		with open("car_count.csv", 'a') as file:
			file.write(', '.join([image_path, str(car_count)]) + '\n')
		io.imsave(os.path.join("car_predictions", "cars_" + image_path), car_image)
