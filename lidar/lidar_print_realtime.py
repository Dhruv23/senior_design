#!/usr/bin/env python3
import time
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from pyrplidar import PyRPlidar

PORT = "/dev/ttyUSB0"
BAUD = 115200
CONNECT_TIMEOUT = 8
MOTOR_PWM = 500
FORCE_SCAN_RETRIES = 3
RETRY_DELAY = 1.0
CSV_PATH = "lidar_recent.csv"
BATCH_SIZE = 1000  # number of points before overwriting CSV


def realtime_plot_with_csv():
    """Continuously read LiDAR data, plot in real time, and log last 1000 points to CSV."""
    try:
        print(f"Attempting connect to {PORT} at {BAUD} baud...")
        lidar = PyRPlidar()
        lidar.connect(port=PORT, baudrate=BAUD, timeout=CONNECT_TIMEOUT)
        print("PyRPlidar Info: device is connected")
    except Exception as e:
        print("Failed to connect:", e)
        return

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

        # --- Start scanning ---
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
            print("Unable to start force_scan(). Check library version.")
            return

        # --- Setup live plot ---
        plt.ion()
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        scatter = ax.scatter([], [], s=10, c="b", alpha=0.6)
        ax.set_title("RPLidar Live Scan (Polar View)")
        ax.set_ylim(0, 6000)
        plt.show(block=False)

        print("Streaming live LiDAR data... Press Ctrl+C to stop.")

        angles, distances = [], []
        data_buffer = []  # store last 1000 points for CSV
        last_update = time.time()
        frame_interval = 0.05  # seconds between redraws

        for scan in gen:
            scan_dict = vars(scan)
            distance = scan_dict.get("distance", 0)
            angle = scan_dict.get("angle", 0)
            quality = scan_dict.get("quality", 0)
            start_flag = scan_dict.get("start_flag", False)

            if distance <= 0:
                continue

            angle_rad = math.radians(angle)
            angles.append(angle_rad)
            distances.append(distance)

            # Append to rolling CSV buffer
            data_buffer.append({
                "angle": angle,
                "distance": distance,
                "quality": quality,
                "start_flag": start_flag
            })

            # If buffer reaches 1000 points → save and clear
            if len(data_buffer) >= BATCH_SIZE:
                df = pd.DataFrame(data_buffer)
                df.to_csv(CSV_PATH, index=False)
                print(f"[Saved] {len(data_buffer)} points → {CSV_PATH}")
                data_buffer.clear()

            # Update plot every ~50ms
            if time.time() - last_update > frame_interval:
                scatter.set_offsets([[a, d] for a, d in zip(angles, distances)])
                ax.set_ylim(0, max(distances) + 500)
                plt.draw()
                plt.pause(0.001)
                last_update = time.time()

            # Reset arrays when a full rotation completes
            if angle < 5 and len(angles) > 20:
                angles.clear()
                distances.clear()

    except KeyboardInterrupt:
        print("\nUser interrupted. Stopping...")
    except Exception as e:
        print("Error during scan:", e)
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
        plt.ioff()
        plt.close("all")
        print("LiDAR stopped and disconnected cleanly.")


if __name__ == "__main__":
    realtime_plot_with_csv()
