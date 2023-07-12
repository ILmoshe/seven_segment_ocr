import cv2
from numpy.typing import NDArray
import numpy as np


def apply_greyscale(img: NDArray) -> NDArray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_contrast(
    img: NDArray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> NDArray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)
    return enhanced


def brighten_image(image):
    threshold = 90
    mask = np.where(image > threshold, 1, 0)

    # Add 20 to pixels that satisfy the mask
    brightened_image = image + (60 * mask)
    brightened_image = np.clip(brightened_image, 0, 255)
    brightened_image = brightened_image.astype(np.uint8)
    return brightened_image


def apply_threshold(image):
    _, thresholded = cv2.threshold(brighten_image(image), 110, 255, cv2.THRESH_BINARY)
    return thresholded


def apply_filters(img: NDArray) -> NDArray:
    gray = apply_greyscale(img)
    enhanced = apply_contrast(gray)
    image_with_filters = apply_threshold(enhanced)
    # plt.imshow(image_after_trash, cmap="gray")
    # plt.title("Thresholded Image")
    # plt.show()
    return image_with_filters


# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.subplot(1, 2, 2)
