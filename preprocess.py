import cv2
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np

# read the input image
image = cv2.imread('fullview.jpg')


def apply_greyscale(img: NDArray) -> NDArray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_contrast(img: NDArray, clip_limit: float = 2., tile_grid_size: tuple = (8, 8)) -> NDArray:
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
    _, thresholded = cv2.threshold(brighten_image(enhanced), 110, 255, cv2.THRESH_BINARY)
    return thresholded


gray = apply_greyscale(image)
enhanced = apply_contrast(gray)
image_after_trash = apply_threshold(enhanced)

# adjusted = apply_contrast(image)


# bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
#
# # Add the original and filtered image to enhance clarity
# clarity = cv2.addWeighted(enhanced, 1.5, bilateral, -0.5, 0)


# laplacian = cv2.Laplacian(clarity, cv2.CV_64F)
# texture = np.uint8(np.absolute(laplacian))

# image_float = clarity.astype(np.float32) / 255.0
#
# # Add brightness by increasing pixel values
# brightened = image_float + 0.1
#
# # Clip the pixel values to ensure they remain within the valid range of 0-1
# brightened = np.clip(brightened, 0, 1)
#
# # Convert the image back to the unsigned 8-bit integer format
# brightened = (brightened * 255).astype(np.uint8)


# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.subplot(1, 2, 2)
plt.imshow(image_after_trash, cmap="gray")
plt.title("Thresholded Image")
plt.show()

# cv2.imshow('original', image)
# cv2.imshow('adjusted', brighten_image(enhanced))
# cv2.waitKey()
# cv2.destroyAllWindows()
