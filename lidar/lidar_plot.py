#!/usr/bin/env python3
import time
import sys
import math
import matplotlib.pyplot as plt
from pyrplidar import PyRPlidar

PORT = "/dev/ttyUSB0"
BAUD = 115200
CONNECT_TIMEOUT = 8
MOTOR_PWM = 500
MAX_POINTS = 400  # number of plotted samples
FORCE_SCAN_RETRIES = 3
RETRY_DELAY = 1.0

def main():
    try:
        print(f"Attempting connect to {PORT} at {BAUD} baud...")
        lidar = PyRPlidar()
        lidar.connect(port=PORT, baudrate=BAUD, timeout=CONNECT_TIMEOUT)
        print("PyRPlidar Info: device is connected")
    except Exception as e:
        print("Failed to connect:", e)
        return 1

    try:
        lidar.set_motor_pwm(MOTOR_PWM)
        print("Motor PWM set. Waiting 2s for spin-up...")
        time.sleep(2)

        try:
            print("Device Info:", lidar.get_info())
        except Exception:
            pass
        try:
            print("Health:", lidar.get_health())
        except Exception:
            pass

        # Try to start force_scan
        gen = None
        for attempt in range(1, FORCE_SCAN_RETRIES + 1):
            try:
                print(f"Starting force_scan() (attempt {attempt}) ...")
                scan_generator = lidar.force_scan()
                gen = scan_generator()
                break
            except Exception as e:
                print(f"force_scan() attempt {attempt} failed: {e}")
                time.sleep(RETRY_DELAY)

        if gen is None:
            print("Unable to start force_scan(). Check library version or inspect available methods.")
            print("Available PyRPlidar attributes:", sorted([a for a in dir(lidar) if not a.startswith('_')]) )
            return 2

        # --- Visualization Setup ---
        plt.ion()
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        scatter = ax.scatter([], [], s=5, c='b', alpha=0.7)
        ax.set_ylim(0, 6000)  # in mm
        ax.set_title("RPLidar Live Scan")
        plt.show(block=False)

        angles = []
        distances = []

        for count, scan in enumerate(gen):
            if not isinstance(scan, dict):
                print("Unexpected scan format:", scan)
                continue

            angle = scan.get("angle", 0.0)
            distance = scan.get("distance", 0.0)
            quality = scan.get("quality", 0)
            start_flag = scan.get("start_flag", False)

            if distance <= 0:
                continue  # skip invalid readings

            # Convert angle to radians for polar plot
            angle_rad = math.radians(angle)

            angles.append(angle_rad)
            distances.append(distance)

            # Periodically update the plot
            if count % 20 == 0:
                scatter.set_offsets(
                    [[a, d] for a, d in zip(angles, distances)]
                )
                ax.set_ylim(0, max(distances) + 500)
                plt.draw()
                plt.pause(0.001)

            if count >= MAX_POINTS:
                break

        print("Scan complete. Close the plot window to exit.")
        plt.ioff()
        plt.show()

    except Exception as e:
        print("Unhandled exception during scan:", repr(e))
        raise
    finally:
        try:
            lidar.stop()
        except Exception:
            pass
        try:
            lidar.set_motor_pwm(0)
        except Exception:
            pass
        lidar.disconnect()
        print("PyRPlidar Info: device is disconnected")
        print("Disconnected cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
