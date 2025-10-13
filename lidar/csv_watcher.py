#!/usr/bin/env python3
import os
import time
import pandas as pd

CSV_PATH = "lidar_recent.csv"
CHECK_INTERVAL = 1.0  # seconds between file checks

def read_csv_as_string(path: str) -> str:
    """Read CSV and return all rows as one string with ^ separators."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return ""
        # Convert each row to a space-separated string, then join with ^
        rows = df.astype(str).apply(lambda row: " ".join(row.values), axis=1)
        combined = "^".join(rows)
        return combined
    except Exception as e:
        return f"Error reading {path}: {e}"

def watch_csv(path: str):
    """Watch the given CSV file and print its contents when it changes."""
    print(f"Watching {path} for updates...")
    last_mtime = None

    while True:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if last_mtime is None or mtime != last_mtime:
                last_mtime = mtime
                output = read_csv_as_string(path)
                if output:
                    print("\n--- CSV Updated ---")
                    print(output)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    watch_csv(CSV_PATH)
