import numpy as np
from skimage import io
from segmentation import segment, padded_segments
from keras.models import load_model
from argparse import ArgumentParser
import os

default_threshold = 0.3

if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("image_path", default="input.jpg")
	parser.add_argument("--model", type=int, default=0)
	parser.add_argument("--threshold", type=float, default=default_threshold)
	args = parser.parse_args()

	model = load_model(os.path.join("models", f"model_{args.model}"))
	image = io.imread(args.image_path)
	segments = segment(image)
	padded, segment_val = padded_segments(image, segments, list(range(segments.max() + 1)))
	predictions = model.predict(padded)
	
	count = 0
	for val, pred in zip(segment_val, predictions):
		image[segments == val] = [255 * pred[1], 255 * pred[0], 0]
		count += pred[0] > args.threshold

	print(f"Found {count} cars")
	io.imsave(os.path.join("outputs", f"cars_{os.path.basename(args.image_path)}"), image)