import numpy as np
from image_utils import *
from segmentation import *
from keras.models import load_model

def center_of_segment(segments, value):
	mask = mask_from_segments(segments, value)
	positions = mask.nonzero()
	length = len(positions[0])
	sum_x, sum_y = sum(positions[0]), sum(positions[1])
	return (sum_x / length, sum_y / length)

if __name__ == "__main__":

	image = load("input.jpg")
	segments = segment(image)
	padded_segments = np.array(images_from_selection(image, segments, range(segments.max() + 1)))

	model = load_model("model")
	predictions = model.predict(padded_segments)

	count = 0
	for i, pred in enumerate(predictions):
		if pred[0] > pred[1]:
			image[center_of_segment(segments, i)] = [255, 0, 0]
			count += 1

	print(f"Found {count} cars")
	save(image, "output.jpg")