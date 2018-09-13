from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

img_width = 50
img_height = 50
img_depth = 4

def load(path):
	try:
		return [io.imread(os.path.join(path, img_path)) for img_path in os.listdir(path)]
	except NotADirectoryError:
		return io.imread(path)

def save(images, path, names):
	try:
		for image, name in zip(images, names):
			io.imsave(os.path.join(path, f"{name}.jpg"), image)
	except TypeError:
		io.imsave(path, image)


def view(image):
	io.imshow(image)
	plt.show()

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

def image_from_4_channel(img):
	new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			new_img[i, j] = img[i, j, :3]
	return new_img