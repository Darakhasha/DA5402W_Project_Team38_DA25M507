import os
import subprocess
import sys


def main():
    expected = (
        f"{os.environ['IMAGE_NAME']}:{os.environ['IMAGE_TAG']}"
    )

    actual = subprocess.check_output(
        [
            "kubectl",
            "get",
            "deployment",
            "taxi-api",
            "-o",
            "jsonpath={.spec.template.spec.containers[0].image}",
        ],
        text=True,
    ).strip()

    print(f"Expected image: {expected}")
    print(f"Actual image:   {actual}")

    if actual != expected:
        raise RuntimeError(
            f"Deployed image does not match expected image.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print("Deployed image verification passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Verification failed: {exc}")
        sys.exit(1)