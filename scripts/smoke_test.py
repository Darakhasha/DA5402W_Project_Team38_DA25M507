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
        # Retry loop to wait for port-forward and server to be ready
        max_retries = 10
        response = None
        
        for attempt in range(max_retries):
            try:
                time.sleep(2)
                response = urllib.request.urlopen(
                    f"http://127.0.0.1:{LOCAL_PORT}/health",
                    timeout=5,
                )
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                print(f"Waiting for port-forward tunnel to open (attempt {attempt + 1})...")

        if response.status != 200:
            raise RuntimeError(
                f"Health check failed. HTTP status: {response.status}"
            )

        print("API health check passed.")

    finally:
        process.terminate()

        try:
            process.wait(timeout=100)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        sys.exit(1)
