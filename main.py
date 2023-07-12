from collections import namedtuple

import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from preprocess import apply_filters
from seven_segment_ocr import get_segments_from_image, identify_digit

Coordinates = namedtuple("Coordinates", ["x", "y"])
DigitsValue = namedtuple("DigitValue", ["left_digit", "right_digit"])


def get_center(img: NDArray) -> NDArray:
    """
    Gets the center of ths screen
    :param img: The Image to crop
    :return: Cropped Image
    """
    # first crop get humidity and temp
    left_up = Coordinates(x=486, y=190)
    right_bottom = Coordinates(x=586, y=585)
    humidity_temp = crop(img, left_up, right_bottom)

    filtered = apply_filters(humidity_temp)
    plt.imshow(filtered, cmap="gray")
    plt.title("cropped & filtered Image")
    plt.show()
    return filtered


def crop(img: NDArray, left_up: Coordinates, right_bottom: Coordinates) -> NDArray:
    cropped_img = img[left_up.y : right_bottom.y, left_up.x : right_bottom.x]
    return cropped_img


def get_temp(img: NDArray) -> DigitsValue:
    """
    Gets the actual value of the temperature digits:
    Actual length came kind of diff, but it doesn't really matter.
    Left digit: 57. Right digit: 55. It can cause a problem
    :param img: Image which are cropped to the temperature digits.
    :return: The actual two digit value
    """
    left_up_left_digit = Coordinates(x=0, y=0)
    right_bottom_left_digit = Coordinates(x=96, y=58)
    left_digit = crop(img, left_up_left_digit, right_bottom_left_digit)

    left_up_right_digit = Coordinates(x=1, y=70)
    right_bottom_right_digit = Coordinates(x=96, y=126)
    right_digit = crop(img, left_up_right_digit, right_bottom_right_digit)

    predicted_left_digit = predict(left_digit)
    predicted_right_digit = predict(right_digit)

    plt.imshow(left_digit, cmap="gray")
    plt.title("left_digit")
    plt.show()
    plt.imshow(right_digit, cmap="gray")
    plt.title("right_digit")
    plt.show()

    return DigitsValue(
        left_digit=predicted_left_digit, right_digit=predicted_right_digit
    )


def get_humidity(img: NDArray) -> DigitsValue:
    """
    Gets the actual value of the temperature digits:
    Actual length came kind of diff, but it doesn't really matter.
    Left digit: 57. Right digit: 55.
    :param img: Image which are cropped to the temperature digits.
    :return: The actual two digit value
    """
    left_up_left_digit = Coordinates(x=0, y=0)
    right_bottom_left_digit = Coordinates(x=96, y=58)
    left_digit = crop(img, left_up_left_digit, right_bottom_left_digit)
    left_up_right_digit = Coordinates(x=1, y=70)
    right_bottom_right_digit = Coordinates(x=96, y=126)
    right_digit = crop(img, left_up_right_digit, right_bottom_right_digit)
    plt.imshow(left_digit, cmap="gray")
    plt.title("left_digit")
    plt.show()
    plt.imshow(right_digit, cmap="gray")
    plt.title("right_digit")
    plt.show()


def get_digit():
    pass


def predict(img: NDArray) -> int:
    segments = get_segments_from_image(img)
    digit = identify_digit(segments)
    return digit


image = cv2.imread("fullview.jpg")
cropped = get_center(image)
get_temp(cropped)
