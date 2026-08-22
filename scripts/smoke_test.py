import subprocess
import sys
import time
import urllib.request


LOCAL_PORT = 18000
SERVICE_PORT = 8000


def main():

    print("Starting smoke test...")
    print("Port-forwarding taxi-api-service...")

    process = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "service/taxi-api-service",
            f"{LOCAL_PORT}:{SERVICE_PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:

        # Give kubectl time to start
        print("Waiting for kubectl port-forward to start...")

        for i in range(30):

            # Check whether kubectl died
            if process.poll() is not None:
                stdout, stderr = process.communicate()

                raise RuntimeError(
                    "kubectl port-forward terminated unexpectedly.\n"
                    f"STDOUT:\n{stdout}\n"
                    f"STDERR:\n{stderr}"
                )

            try:

                response = urllib.request.urlopen(
                    f"http://127.0.0.1:{LOCAL_PORT}/health",
                    timeout=5,
                )

                status = response.status

                print(
                    f"Health endpoint responded with HTTP {status}"
                )

                if status == 200:
                    print("API health check passed.")
                    return

                raise RuntimeError(
                    f"Health endpoint returned HTTP {status}"
                )

            except Exception as exc:

                print(
                    f"Attempt {i + 1}/30: API not ready yet "
                    f"({exc})"
                )

                time.sleep(2)

        # If we reach here, all attempts failed
        stdout, stderr = process.communicate(timeout=5)

        raise RuntimeError(
            "Smoke test timed out after 60 seconds.\n"
            f"Port-forward STDOUT:\n{stdout}\n"
            f"Port-forward STDERR:\n{stderr}"
        )

    finally:

        print("Stopping port-forward...")

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