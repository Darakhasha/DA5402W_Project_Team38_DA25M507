import argparse
import random
import time

import pandas as pd
import requests


parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    default="http://localhost:8000/predict",
)

parser.add_argument(
    "--iterations",
    type=int,
    default=10,
)

parser.add_argument(
    "--mode",
    choices=["normal", "drift"],
    default="normal",
)

parser.add_argument(
    "--reference",
    default="dummy/training_data.csv",
)

args = parser.parse_args()


reference = pd.read_csv(
    args.reference
)


# These are not model input features.
EXCLUDED_COLUMNS = {
    "timestamp",
    "taxi_demand",
}


FEATURE_COLUMNS = [
    column
    for column in reference.columns
    if column not in EXCLUDED_COLUMNS
]


def generate_data(mode):

    row = reference.sample(
        n=1
    ).iloc[0]

    payload = {}

    for column in FEATURE_COLUMNS:

        value = row[column]

        if mode == "normal":

            payload[column] = (
                value.item()
                if hasattr(value, "item")
                else value
            )

        else:

            # Deliberately shift numeric values
            # to create feature drift.
            if pd.api.types.is_numeric_dtype(
                reference[column]
            ):

                mean = reference[
                    column
                ].mean()

                std = reference[
                    column
                ].std()

                if pd.isna(std) or std == 0:
                    std = 1

                payload[column] = float(
                    mean + 5 * std
                )

            else:

                payload[column] = (
                    value.item()
                    if hasattr(value, "item")
                    else value
                )

    return payload


for i in range(args.iterations):

    payload = generate_data(
        args.mode
    )

    response = requests.post(
        args.url,
        json=payload,
        timeout=10,
    )

    response.raise_for_status()

    result = response.json()

    print(
        f"\nIteration "
        f"{i + 1}/{args.iterations}"
    )

    print(
        "Input:",
        payload
    )

    print(
        "Response:",
        result
    )

    time.sleep(1)