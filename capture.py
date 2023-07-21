import csv
import time
from typing import get_type_hints

import cv2
import arrow as Arrow
from numpy.typing import NDArray
from loguru import logger as Logger
import matplotlib.pyplot as plt

from get_digits import get_center, get_humidity, get_temp, get_wind
from preprocess import apply_filters
from save import save_csv, valueSchema


DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        Logger.error("Error: Could not access the webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    Logger.info(f"Staring ......")
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                Logger.error("Error: Failed to capture frame from the webcam")
                break

            cv2.imshow("Webcam", frame)
            predicted: dict = start_calculation(frame)
            save_csv(csv_writer, predicted)

            time.sleep(10)  # its not async so its actually block everything

            exit = cv2.waitKey(1) & 0xFF == ord("q")
            if exit:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def start_calculation(image: NDArray) -> valueSchema:
    output = {}

    filterd_image_for_center = apply_filters(image)
    cropped = get_center(filterd_image_for_center)
    # This function need to get change according to image size
    # plt.imshow(cropped, cmap="gray")
    # plt.title("image")
    # plt.show()

    temperature = get_temp(cropped)
    humidity = get_humidity(cropped)

    filterd_image_for_wind = apply_filters(image, False)
    wind = get_wind(filterd_image_for_wind)

    Logger.info(
        f"temperature {temperature.left_digit}{temperature.right_digit}. humidity {humidity.left_digit}{humidity.right_digit}. wind {wind.left_digit}{wind.right_digit}"
    )

    output["temperature"] = temperature
    output["humidity"] = humidity
    output["wind"] = wind
    output["timestamp"] = Arrow.utcnow().format()
    return output


def fake_calcul():
    frame = cv2.imread("fullview.jpg")
    if frame is None:
        Logger.error("Could'nt fetch image")
        return
    predicted: valueSchema = start_calculation(frame)
    save_csv(csv_writer, predicted)


if __name__ == "__main__":
    with open(
        f"image_data-{Arrow.utcnow().format('X')}.csv", "w", newline=""
    ) as csvfile:
        global csv_writer
        Logger.info(f"{list(get_type_hints(valueSchema))}")
        csv_writer = csv.DictWriter(
            csvfile, fieldnames=list(get_type_hints(valueSchema))
        )
        csv_writer.writeheader()

        capture_image()
