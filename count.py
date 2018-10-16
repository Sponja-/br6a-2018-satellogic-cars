import numpy as np
from skimage import io
from segmentation import segment, padded_image
from keras.models import load_model
from argparse import ArgumentParser
import os

if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("image_path", default="input.jpg")
	parser.add_argument("--model", type=int, default=0)
	args = parser.parse_args()

	image = io.imread(args.image_path)
	segments = segment(image)

	padded_segments = []
	segment_val = []
	max_val = segments.max() + 1
	for i in range(max_val):
		img = padded_image(image, segments, i)
		if img is not None:
			padded_segments.append(img)
			segment_val.append(i)
		print(f"Padding images [{int((i / max_val) * 100)}%]\r", end="")

	padded_segments = np.array(padded_segments)

	model = load_model(os.path.join("models", f"model_{args.model}"))
	predictions = model.predict(padded_segments)

	count = 0
	for i, pred in zip(segment_val, predictions):
		image[segments == i] = [255 * pred[1], 255 * pred[0], 0]
		count += pred[0] > 0.3

	print(f"Found {count} cars")
	io.imsave(os.path.join("outputs", f"cars_{os.path.basename(args.image_path)}"), image)