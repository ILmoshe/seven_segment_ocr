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


def get_segments_from_image(image_path) -> list[int]:
    image = cv2.imread(image_path)

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # visulize(image, thresholded)

    # Segment dimensions and positions. make it dynamic latter
    segment_width = 10
    segment_height = 10
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
        segment: NDArray = thresholded[
            h_start : h_start + segment_height, w_start : w_start + segment_width
        ]
        segment_mean = np.mean(segment)
        segment_containts_black_pixels = segment_mean > 150
        segments.append(1) if segment_containts_black_pixels else segments.append(0)

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


print(f"segmenting number 2 _-----------------------------------")
image_path = "2.jpg"
segments = get_segments_from_image(image_path)
print(segments)
digit = identify_digit(segments)
print("digit:", digit)


print(f"segmenting number 5 _-----------------------------------")
image_path = "5.jpg"
segments = get_segments_from_image(image_path)
print(segments)
digit = identify_digit(segments)
print("digit:", digit)