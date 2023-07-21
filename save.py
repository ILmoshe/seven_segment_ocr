from typing import TypedDict
from csv import DictWriter

from get_digits import DigitsValue


class valueSchema(TypedDict):
    temperature: DigitsValue
    humidity: DigitsValue
    wind: DigitsValue
    timestamp: str


def save_csv(csv_writer: DictWriter, predicted: valueSchema) -> bool:
    predicted[
        "humidity"
    ] = f"{predicted['humidity'].left_digit}{predicted['humidity'].right_digit}"
    predicted[
        "temperature"
    ] = f"{predicted['temperature'].left_digit}{predicted['temperature'].right_digit}"
    predicted["wind"] = f"{predicted['wind'].left_digit}{predicted['wind'].right_digit}"
    csv_writer.writerow(predicted)
