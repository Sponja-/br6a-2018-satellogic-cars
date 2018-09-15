from __future__ import division
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from image_utils import *

_selected_segments = set()
_current_segments = []
_current_image = []
_original_image = []
_plt_img = []
_shift = False


def segment(image, **kwargs):
	return slic(img_as_float(image), n_segments = kwargs.get("n_segments", max(image.shape) * 1.5), sigma = 5)

def on_click(event):
	if _shift:
		x, y = int(round(event.xdata)), int(round(event.ydata))
		segment_value = _current_segments[y, x]

		if segment_value not in _selected_segments:
			_selected_segments.add(segment_value)
			_current_image[_current_segments == segment_value] = [255, 0, 0]
		else:
			_selected_segments.remove(segment_value)
			_current_image[_current_segments == segment_value] = _original_image[_current_segments == segment_value]
		_plt_img.set_data(_current_image)
		plt.draw()
		print(segment_value)

def on_key_press(event):
	global _shift
	if event.key == 'shift':
		_shift = True

def on_key_release(event):
	global _shift
	if event.key == 'shift':
		_shift = False

def select(image, segments):
	global _selected_segments
	global _current_segments
	global _current_image
	global _original_image
	global _plt_img

	_selected_segments = set()
	_current_segments = segments
	_current_image = np.copy(image)
	_original_image = image

	fig = plt.figure(f"Segmentation")
	ax = fig.add_subplot(1, 1, 1)
	_plt_img = ax.imshow(image)
	
	fig.canvas.mpl_connect('button_press_event', on_click)
	fig.canvas.mpl_connect('key_press_event', on_key_press)
	fig.canvas.mpl_connect('key_release_event', on_key_release)

	plt.show()

	return _selected_segments

def mask_from_segments(segments, value):
	mask = np.zeros(segments.shape, dtype="uint8")
	mask[segments == value] = 255
	return mask

def mask_not_segments(segments, value):
	mask = np.full(segments.shape, 255, dtype="uint8")
	mask[segments == value] = 0
	return mask

def strip(image):
	gray = image
	if len(image.shape) != 2:
		gray = grayscale(image)
	x, y, width, height = cv2.boundingRect(cv2.findNonZero(gray))
	return image[y:y + height, x:x + width]

def train_image(image, segments, value):

	mask = mask_from_segments(segments, value)

	positions = np.transpose(mask.nonzero())
	x, y, width, height = cv2.boundingRect(positions[:,::-1])

	global_height, global_width, _ = image.shape
	left_padding_x, top_padding_y = (img_width - width) // 2, (img_height - height) // 2
	right_padding_x, bottom_padding_y = left_padding_x, top_padding_y

	if (img_width - width) % 2 == 1:
		right_padding_x += 1
	if (img_height - height) % 2 == 1:
		bottom_padding_y += 1 

	if top_padding_y > y:
		return None
	if left_padding_x > x:
		return None
	if bottom_padding_y > global_height - (y + height):
		return None
	if right_padding_x > global_width - (x + width):
		return None

	result_image = np.zeros((img_height, img_width, 4), dtype="float32")

	# i is result_image's index, ii is original image's index
	for i, ii in zip(range(img_height), range(y - top_padding_y, y + height + bottom_padding_y)):
		for j, jj in zip(range(img_width), range(x - left_padding_x, x + width + right_padding_x)):
			# Add a channel to whether each pixel belongs to the original segment
			result_image[i, j] = np.array(list(image[ii, jj]) + [mask[ii, jj]], dtype="float32")

	# returns a 4-channel image with dimensions (image_utils.img_width x image_utils.img_height)
	return result_image


"""
	# Was slower

	result_image = np.zeros((global_height, global_width, 4))
	# Add a channel to represent whether each pixel belongs to the original segment
	for i in range(global_height):
		for j in range(global_width):
			result_image[i, j] = np.array(list(image[i, j]) + [mask[i, j]])


	
	return result_image[(y - top_padding_y):(y + height + bottom_padding_y),(x - left_padding_x):(x + width + right_padding_x)].astype("float32")

"""
def images_from_selection(image, segments, selection):
	result = []
	print(f"{len(selection)} segments")
	for i, val in enumerate(selection):
		print(i + 1)
		train = train_image(image, segments, val)
		# train_image returns None when it can't produce an image_utils.img_width x image_utils.img_height image
		if train is not None:
			result.append(train) 
	return result

if __name__ == "__main__":

	images = load("inputs")
	image_paths = os.listdir("inputs")
	print(f"Found {len(images)} inputs")

	true_index = len(os.listdir("true_segments"))
	false_index = len(os.listdir("false_segments"))

	print("Segmenting")
	segments = [segment(image) for image in images]

	for i in range(len(images)):

		selection = select(images[i], segments[i])

		true_train_images = images_from_selection(images[i], segments[i], selection)
		
		print(f"Saving {len(true_train_images)} car images")
		for img in true_train_images:
			# Can't save it as an image: it has an extra channel
			with open(os.path.join("data", "car_" + str(true_index)), 'wb') as save_file:
				np.save(save_file, img)
			true_index += 1

		not_selection = set(range(segments[i].max())) - selection
		false_train_images = images_from_selection(images[i], segments[i], not_selection)

		print(f"Saving {len(false_train_images)} non-car images")
		for img in false_train_images:
			with open(os.path.join("data", str(false_index)), 'wb') as save_file:
				np.save(save_file, img)
			false_index += 1
		
		os.rename(os.path.join("inputs", image_paths[i]), os.path.join("processed_inputs", image_paths[i]))