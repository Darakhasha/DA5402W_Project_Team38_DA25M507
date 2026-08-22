import subprocess
import sys
import time
import urllib.request
import urllib.error


LOCAL_PORT = 18000
SERVICE_PORT = 8000
MAX_ATTEMPTS = 60
WAIT_SECONDS = 2


def main():

    print("=== SMOKE TEST START ===")

    # --------------------------------------------------
    # 1. Show Kubernetes state
    # --------------------------------------------------

    print("\n=== PODS ===")

    subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-l",
            "app=taxi-api",
            "-o",
            "wide",
        ],
        check=False,
    )

    print("\n=== SERVICE ===")

    subprocess.run(
        [
            "kubectl",
            "get",
            "svc",
            "taxi-api-service",
        ],
        check=False,
    )

    print("\n=== ENDPOINTS ===")

    subprocess.run(
        [
            "kubectl",
            "get",
            "endpoints",
            "taxi-api-service",
        ],
        check=False,
    )

    # --------------------------------------------------
    # 2. Start kubectl port-forward
    # --------------------------------------------------

    print("\n=== STARTING PORT FORWARD ===")

    port_forward = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "service/taxi-api-service",
            f"{LOCAL_PORT}:{SERVICE_PORT}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(
        f"Port forwarding started: "
        f"localhost:{LOCAL_PORT} -> service:{SERVICE_PORT}"
    )

    try:

        # --------------------------------------------------
        # 3. Give kubectl a moment to establish forwarding
        # --------------------------------------------------

        time.sleep(3)

        # --------------------------------------------------
        # 4. Wait for /health
        # --------------------------------------------------

        url = f"http://127.0.0.1:{LOCAL_PORT}/health"

        print(f"\nWaiting for API: {url}")

        for attempt in range(1, MAX_ATTEMPTS + 1):

            # Check whether kubectl died
            if port_forward.poll() is not None:

                print(
                    "\nERROR: kubectl port-forward "
                    "process stopped unexpectedly."
                )

                raise RuntimeError(
                    "kubectl port-forward exited before API became ready."
                )

            try:

                with urllib.request.urlopen(
                    url,
                    timeout=5,
                ) as response:

                    status = response.status

                    print(
                        f"Attempt {attempt}/{MAX_ATTEMPTS}: "
                        f"HTTP {status}"
                    )

                    if status == 200:

                        print("\n================================")
                        print("SMOKE TEST PASSED")
                        print("================================")

                        return

            except urllib.error.HTTPError as exc:

                print(
                    f"Attempt {attempt}/{MAX_ATTEMPTS}: "
                    f"HTTP {exc.code}"
                )

            except urllib.error.URLError as exc:

                print(
                    f"Attempt {attempt}/{MAX_ATTEMPTS}: "
                    f"connection not ready ({exc.reason})"
                )

            except Exception as exc:

                print(
                    f"Attempt {attempt}/{MAX_ATTEMPTS}: "
                    f"{type(exc).__name__}: {exc}"
                )

            time.sleep(WAIT_SECONDS)

        # --------------------------------------------------
        # 5. Timeout
        # --------------------------------------------------

        raise RuntimeError(
            f"API did not become ready after "
            f"{MAX_ATTEMPTS * WAIT_SECONDS} seconds."
        )

    finally:

        # --------------------------------------------------
        # 6. Always stop port-forward
        # --------------------------------------------------

        print("\nStopping port-forward...")

        if port_forward.poll() is None:

            port_forward.terminate()

            try:
                port_forward.wait(timeout=5)

            except subprocess.TimeoutExpired:

                print("Force stopping kubectl...")

                port_forward.kill()
                port_forward.wait()


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"\nSmoke test failed: "
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)