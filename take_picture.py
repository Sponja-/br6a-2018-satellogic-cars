from pyautogui import screenshot
import os

image_index = len(os.listdir("inputs")) + len(os.listdir("processed")) + 1
screenshot(os.path.join("inputs", f"{image_index}.jpg"), region = (374, 81, 829, 708))