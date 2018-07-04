from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

_selected_segments = set()
_current_segments = []
_current_image = []
_original_image = []
_plt_img = []
_shift = False

def load(path):
	return io.imread(path)

def load_all(path):
	return [io.imread(os.path.join(path, img_path)) for img_path in os.listdir(path)]

def save(image, path):
	io.imsave(path, image)

def save_all(images, path):
	for i, image in enumerate(images):
		io.imsave(os.path.join(path, f"{i}.jpg"), image)

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
		_plt_img.set_data(mark_boundaries(_current_image, _current_segments))
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
	_plt_img = ax.imshow(mark_boundaries(image, segments))
	
	fig.canvas.mpl_connect('button_press_event', on_click)
	fig.canvas.mpl_connect('key_press_event', on_key_press)
	fig.canvas.mpl_connect('key_release_event', on_key_release)

	plt.show()

	return np.array(list(_selected_segments))

def view(image):
	io.imshow(image)
	plt.show()

def mask_from_segments(segments, values):
	mask = np.zeros(segments.shape, dtype="uint8")
	for value in values:
		mask[segments == value] = 255
	return mask

def mask_not_segments(segments, values):
	mask = np.full(segments.shape, 255, dtype="uint8")
	for value in values:
		mask[segments == value] = 0
	return mask

def apply_mask(image, mask):
	return cv2.bitwise_and(image, image, mask=mask)

def grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def strip(image):
	gray = image
	if len(image.shape) != 2:
		gray = grayscale(image)
	x, y, width, height = cv2.boundingRect(cv2.findNonZero(gray))
	return image[y:y + height, x:x + width]