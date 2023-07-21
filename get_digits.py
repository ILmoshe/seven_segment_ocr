from collections import namedtuple
from dataclasses import dataclass
from typing import TypedDict

from loguru import logger as Logger
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
    coord = DigitCoordinates(
        left_up=Coordinates(x=486, y=190), right_down=Coordinates(x=586, y=585)
    )
    humidity_and_temp = crop(img, coord)

    # plt.imshow(humidity_and_temp, cmap="gray")
    # plt.title("humidity_and_temp Image")
    # plt.show()
    return humidity_and_temp


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
    left_digit = crop(img, digit.left_digit)
    right_digit = crop(img, digit.right_digit)

    predicted_left_digit = predict(left_digit)
    predicted_right_digit = predict(right_digit)
    Logger.info(f"Predicted {predicted_left_digit}{predicted_right_digit}")

    return DigitsValue(
        left_digit=predicted_left_digit, right_digit=predicted_right_digit
    )


def get_wind(img: NDArray) -> DigitsValue:
    initial_crop = DigitCoordinates(
        left_up=Coordinates(x=230, y=161),
        right_down=Coordinates(
            x=421, y=624
        ),  # This thing need to change according to our wanted size
    )
    img = crop(img, initial_crop)
    digit_crop = DigitCoordinates(
        left_up=Coordinates(x=57, y=205), right_down=Coordinates(x=126, y=305)
    )
    img = crop(img, digit_crop)

    left = DigitCoordinates(
        left_up=Coordinates(x=0, y=0), right_down=Coordinates(x=69, y=39)
    )

    right = DigitCoordinates(
        left_up=Coordinates(x=0, y=59), right_down=Coordinates(x=69, y=100)
    )

    digits_coord = TwoDigitCoordinate(left_digit=left, right_digit=right)
    return process_two_digits(img, digits_coord)


def get_temp(img: NDArray) -> DigitsValue:
    left = DigitCoordinates(
        left_up=Coordinates(x=0, y=0), right_down=Coordinates(x=96, y=58)
    )
    right = DigitCoordinates(
        left_up=Coordinates(x=1, y=70), right_down=Coordinates(x=96, y=126)
    )
    digits_coord = TwoDigitCoordinate(left_digit=left, right_digit=right)
    return process_two_digits(img, digits_coord)


def get_humidity(img: NDArray) -> DigitsValue:
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
