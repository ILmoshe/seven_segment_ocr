import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def visulize(image, thresholded):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(thresholded, cmap="gray")
    plt.title("Thresholded Image")
    plt.show()


def get_segments_from_image(image: NDArray) -> list[int]:
    """
    Need to make it dynamic so If I got diff digit size, I would still be able to predict
    :param image:
    :return: The seven segments
    """
    segment_width = 12
    segment_height = 12

    image = cv2.resize(image, (71, 41))
    image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    # plt.imshow(image, cmap="gray")
    # plt.title("image")
    # plt.show()

    positions: list[tuple[int, int]] = [
        (0, 15),
        (15, 0),
        (15, 30),
        (30, 15),
        (45, 0),
        (45, 30),
        (60, 15),
    ]

    segments = []
    for position in positions:
        h_start, w_start = position
        segment: NDArray = image[
            h_start : h_start + segment_height, w_start : w_start + segment_width
        ]
        segment_average = np.average(segment)
        segment_contains_black_pixels = segment_average < 170
        segments.append(1) if segment_contains_black_pixels else segments.append(0)

    return segments


def identify_digit(segments):
    patterns = {
        "0": [1, 1, 1, 0, 1, 1, 1],
        "1": [0, 0, 1, 0, 0, 1, 0],
        "2": [1, 0, 1, 1, 1, 0, 1],
        "3": [1, 0, 1, 1, 0, 1, 1],
        "4": [0, 1, 1, 1, 0, 1, 0],
        "5": [1, 1, 0, 1, 0, 1, 1],
        "6": [1, 1, 0, 1, 1, 1, 1],
        "7": [1, 0, 1, 0, 0, 1, 0],
        "8": [1, 1, 1, 1, 1, 1, 1],
        "9": [1, 1, 1, 1, 0, 1, 1],
    }

    for digit, pattern in patterns.items():
        if segments == pattern:
            return digit

    return None
