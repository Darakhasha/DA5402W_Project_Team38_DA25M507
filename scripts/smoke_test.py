import subprocess
import sys
import time
import urllib.request


LOCAL_PORT = 18000


def get_pod():
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-l",
            "app=taxi-api",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    pod = result.stdout.strip()

    if not pod:
        raise RuntimeError("No taxi-api pod found.")

    return pod


def main():
    pod = get_pod()

    print(f"Testing pod: {pod}")

    process = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            f"pod/{pod}",
            f"{LOCAL_PORT}:8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Give kubectl time to establish the port forward
        time.sleep(5)

        response = urllib.request.urlopen(
            f"http://127.0.0.1:{LOCAL_PORT}/health",
            timeout=10,
        )

        if response.status != 200:
            raise RuntimeError(
                f"Health check failed. HTTP status: {response.status}"
            )

        print("API health check passed.")

    finally:
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        sys.exit(1)