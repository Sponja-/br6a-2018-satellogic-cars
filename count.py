import numpy as np
from image_utils import *
from segmentation import *
from keras.models import load_model
import sys

if __name__ == "__main__":

	if len(sys.argv) != 2:
		image = load("input.jpg")
	else:
		image = load(sys.argv[1])

	segments = segment(image)

	padded_segments = []
	segment_val = []
	for i in range(segments.max() + 1):
		img = padded_image(image, segments, i)
		if img is not None:
			padded_segments.append(img)
			segment_val.append(i)
	padded_segments = np.array(padded_segments)

	model = load_model("model")
	predictions = model.predict(padded_segments)

	count = 0
	for i, pred in zip(segment_val, predictions):
		if pred[0] > pred[1]:
			image[segments == i] = [255, 0, 0]
			count += 1

	print(f"Found {count} cars")
	save(image, "output.jpg")