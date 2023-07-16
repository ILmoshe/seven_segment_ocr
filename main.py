from collections import namedtuple
from dataclasses import dataclass
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from preprocess import apply_filters
from seven_segment_ocr import get_segments_from_image, identify_digit

Coordinates = namedtuple("Coordinates", ["x", "y"])
DigitsValue = namedtuple("DigitValue", ["left_digit", "right_digit"])


class DigitCoordinates(TypedDict):
    left_up: Coordinates
    right_down: Coordinates


@dataclass
class TwoDigitCoordinate:
    left_digit: DigitCoordinates
    right_digit: DigitCoordinates


def get_center(img: NDArray) -> NDArray:
    """
    Gets the center of ths screen
    :param img: The Image to crop
    :return: Cropped Image
    """
    # first crop get humidity and temp
    coord = DigitCoordinates(
        left_up=Coordinates(x=486, y=190), right_down=Coordinates(x=586, y=585)
    )
    humidity_and_temp = crop(img, coord)

    filtered = apply_filters(humidity_and_temp)
    plt.imshow(filtered, cmap="gray")
    plt.title("cropped & filtered Image")
    plt.show()
    return filtered


def get_wind(img: NDArray) -> DigitsValue:
    pass


def crop(img: NDArray, digit: DigitCoordinates) -> NDArray:
    cropped_img = img[
        digit["left_up"].y : digit["right_down"].y,
        digit["left_up"].x : digit["right_down"].x,
    ]
    return cropped_img


def process_two_digits(img: NDArray, digit: TwoDigitCoordinate) -> DigitsValue:
    """
    :param img: Image which are cropped to the temperature digits.
    :param digit:
    :return: The actual two digit value
    """

    left_digit_to_crop = DigitCoordinates(
        left_up=digit.left_digit["left_up"], right_down=digit.left_digit["right_down"]
    )
    left_digit = crop(img, left_digit_to_crop)

    right_digit_to_crop = DigitCoordinates(
        left_up=digit.right_digit["left_up"], right_down=digit.right_digit["right_down"]
    )
    right_digit = crop(img, right_digit_to_crop)

    predicted_left_digit = predict(left_digit)
    predicted_right_digit = predict(right_digit)
    print(f"Predicted {predicted_left_digit}{predicted_right_digit}")

    return DigitsValue(
        left_digit=predicted_left_digit, right_digit=predicted_right_digit
    )


def get_temp(img: NDArray) -> DigitsValue:
    """
    :param img: Image which are cropped to the temperature digits.
    :return: The actual two digit value
    """
    left = DigitCoordinates(
        left_up=Coordinates(x=0, y=0), right_down=Coordinates(x=96, y=58)
    )
    right = DigitCoordinates(
        left_up=Coordinates(x=1, y=70), right_down=Coordinates(x=96, y=126)
    )
    digits_coord = TwoDigitCoordinate(left_digit=left, right_digit=right)
    return process_two_digits(img, digits_coord)


def get_humidity(img: NDArray) -> DigitsValue:
    """
    :param img: Image which are cropped to the temperature digits.
    :return: The actual two digit value
    """

    first_crop = DigitCoordinates(
        left_up=Coordinates(x=1, y=278), right_down=Coordinates(x=97, y=394)
    )
    img = crop(img, first_crop)

    left = DigitCoordinates(
        left_up=Coordinates(x=0, y=0), right_down=Coordinates(x=96, y=54)
    )
    right = DigitCoordinates(
        left_up=Coordinates(x=1, y=70), right_down=Coordinates(x=96, y=126)
    )
    digits_coord = TwoDigitCoordinate(left_digit=left, right_digit=right)
    return process_two_digits(img, digits_coord)


def predict(img: NDArray) -> int:
    segments = get_segments_from_image(img)
    digit = identify_digit(segments)
    return digit


image = cv2.imread("fullview.jpg")
cropped = get_center(image)
temperature = get_temp(cropped)
humidity = get_humidity(cropped)
print(
    f"temperature {temperature.left_digit}{temperature.right_digit}. humidity {humidity.left_digit}{humidity.right_digit}"
)
