import subprocess
import sys
import time
import urllib.request


LOCAL_PORT = 18000
SERVICE_PORT = 8000


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
    # 2. Start port-forward
    # --------------------------------------------------

    print("\n=== STARTING PORT FORWARD ===")

    process = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "service/taxi-api-service",
            f"{LOCAL_PORT}:{SERVICE_PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:

        # --------------------------------------------------
        # 3. Wait for port-forward to become available
        # --------------------------------------------------

        print(
            f"Waiting for port-forward "
            f"127.0.0.1:{LOCAL_PORT}..."
        )

        port_forward_ready = False

        for _ in range(30):

            if process.poll() is not None:

                output = process.stdout.read()

                print("\n=== PORT FORWARD FAILED ===")
                print(output)

                raise RuntimeError(
                    f"kubectl port-forward exited with "
                    f"code {process.returncode}"
                )

            try:

                # Try the API
                response = urllib.request.urlopen(
                    f"http://127.0.0.1:{LOCAL_PORT}/health",
                    timeout=2,
                )

                print(
                    f"Health endpoint returned "
                    f"HTTP {response.status}"
                )

                if response.status == 200:

                    port_forward_ready = True
                    break

            except Exception as exc:

                print(
                    f"API not ready yet: {exc}"
                )

            time.sleep(2)

        # --------------------------------------------------
        # 4. Final result
        # --------------------------------------------------

        if not port_forward_ready:

            print("\n=== PORT FORWARD OUTPUT ===")

            try:
                output = process.stdout.read()
                print(output)
            except Exception:
                pass

            raise RuntimeError(
                "API did not respond within 60 seconds."
            )

        print("\n=== SMOKE TEST PASSED ===")

    finally:

        print("\nStopping port-forward...")

        process.terminate()

        try:

            process.wait(timeout=10)

        except subprocess.TimeoutExpired:

            process.kill()


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(f"\nSmoke test failed: {exc}")

        sys.exit(1)