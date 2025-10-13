#!/usr/bin/env python3
import time
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from pyrplidar import PyRPlidar


PORT = "/dev/ttyUSB0"
BAUD = 115200      # you already confirmed 115200 works
CONNECT_TIMEOUT = 8
MOTOR_PWM = 500
MAX_POINTS = 1000
FORCE_SCAN_RETRIES = 3
RETRY_DELAY = 1.0


def plot_polar(df: pd.DataFrame):
    """Plot LiDAR data in polar coordinates using matplotlib."""
    if df.empty:
        print("No data to plot.")
        return

    # Remove invalid distances
    df = df[df["distance"] > 0]

    # Convert degrees → radians
    angles = df["angle"].apply(math.radians)
    distances = df["distance"]

    # --- Polar Plot ---
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.scatter(angles, distances, s=10, c="b", alpha=0.7)
    ax.set_title("RPLidar Scan (Polar Plot)")
    ax.set_ylim(0, max(distances) * 1.1)
    plt.show()

def main():
    df = pd.DataFrame(columns=['start_flag', 'quality', 'angle', 'distance'])
    try:
        print(f"Attempting connect to {PORT} at {BAUD} baud...")
        lidar = PyRPlidar()
        lidar.connect(port=PORT, baudrate=BAUD, timeout=CONNECT_TIMEOUT)
        print(f"PyRPlidar Info : device is connected")
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

        # Try to get a working force_scan generator
        gen = None
        for attempt in range(1, FORCE_SCAN_RETRIES + 1):
            try:
                print(f"Starting force_scan() (attempt {attempt}) ...")
                scan_generator = lidar.force_scan()   # DO NOT pass timeout kw
                gen = scan_generator()                # this is how your earlier script did it
                break
            except Exception as e:
                print(f"force_scan() attempt {attempt} failed: {e}")
                time.sleep(RETRY_DELAY)

        if gen is None:
            print("Unable to start force_scan(). Check library version or inspect available methods.")
            # show available methods to help debug
            print("Available PyRPlidar attributes:", sorted([a for a in dir(lidar) if not a.startswith('_')]) )
            return 2
        
        # iterate and print measured points (safe)
        for count, scan in enumerate(gen):
            # scan format depends on library version. Print raw so we can inspect.
            df.loc[len(df)] = {k: v for k, v in vars(scan).items() if k in df.columns}

            print(count, scan)
            if count >= MAX_POINTS - 1:
                break
        

    except IndexError as e:
        print("IndexError while parsing descriptor — serial read probably returned empty bytes.")
        print("Possible causes: wrong baud, timeout or temporary device hiccup.")
        raise
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
        print("PyRPlidar Info : device is disconnected")
        print("Disconnected cleanly.")
    print(df)
    return df

if __name__ == "__main__":
    df = main()
    if isinstance(df, pd.DataFrame):
        plot_polar(df)
