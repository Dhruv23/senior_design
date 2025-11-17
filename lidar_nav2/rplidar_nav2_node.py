#!/usr/bin/env python3
"""
RPLidar A1M8 → /scan publisher (ROS2 Nav2 compatible)
- Uses robust continuous scan loop (no body-size errors)
- Averages multiple revolutions
- Writes live JSON snapshot of the latest scan
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from rplidar import RPLidar
import numpy as np
import math, json, time
from collections import defaultdict
from pathlib import Path

# === Configuration ===
PORT = '/dev/ttyUSB0'
BAUD = 115200           # try 256000 if you ever see "Wrong body size"
ANG_RES_DEG = 1.0
REV_AVG = 2
DIST_MIN_MM = 100
DIST_MAX_MM = 6000
FRAME_ID = 'lidar_link'
JSON_PATH = Path('/tmp/lidar_latest.json')  # live snapshot file

def deg_to_idx(deg):
    return int(round(deg % 360 / ANG_RES_DEG))

def robust_bin_average(frames_by_idx):
    idxs, dists = [], []
    for idx, vals in frames_by_idx.items():
        if not vals:
            continue
        arr = np.array(vals)
        idxs.append(idx)
        dists.append(np.median(arr))
    angs = np.array(idxs) * ANG_RES_DEG
    return angs, np.array(dists)

class RPLidarNode(Node):
    def __init__(self):
        super().__init__('rplidar_nav2_node')
        self.publisher_ = self.create_publisher(LaserScan, '/scan', 10)
        self.ring = []
        self.keep_running = True

        # Connect + spin-up delay
        try:
            self.lidar = RPLidar(PORT, baudrate=BAUD)
            info = self.lidar.get_info()
            health = self.lidar.get_health()
            self.get_logger().info(f"Connected: {info}")
            self.get_logger().info(f"Health: {health}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")
            raise

        self.get_logger().info("Waiting 1.5 s for motor spin-up…")
        time.sleep(1.5)

        self.get_logger().info("Starting scan loop…")
        self.create_timer(0.0, self.scan_loop)  # single async task

    # === Continuous scan loop ===
    def scan_loop(self):
        try:
            for i, scan in enumerate(
                self.lidar.iter_scans(max_buf_meas=2000, min_len=50)
            ):
                if not self.keep_running:
                    break

                # --- bin one revolution ---
                bins = defaultdict(list)
                for q, angle, dist in scan:
                    if dist < DIST_MIN_MM or dist > DIST_MAX_MM:
                        continue
                    bins[deg_to_idx(angle)].append(dist)

                self.ring.append(bins)
                if len(self.ring) > REV_AVG:
                    self.ring.pop(0)

                merged = defaultdict(list)
                for frame in self.ring:
                    for idx, vals in frame.items():
                        merged[idx].extend(vals)

                angs, dists = robust_bin_average(merged)
                if len(angs) == 0:
                    continue

                # --- build LaserScan message ---
                msg = LaserScan()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = FRAME_ID

                msg.angle_min = math.radians(float(np.min(angs)))
                msg.angle_max = math.radians(float(np.max(angs)))
                msg.angle_increment = math.radians(ANG_RES_DEG)
                msg.range_min = DIST_MIN_MM / 1000.0
                msg.range_max = DIST_MAX_MM / 1000.0
                msg.scan_time = 1 / 5.5
                msg.time_increment = msg.scan_time / max(len(angs), 1)
                msg.ranges = (dists / 1000.0).tolist()
                msg.intensities = [1.0] * len(dists)

                self.publisher_.publish(msg)
                self.write_json(i, angs, dists)
                self.get_logger().info(f"Revolution {i}: {len(dists)} pts published")

        except Exception as e:
            self.get_logger().error(f"LIDAR error: {e}")
        finally:
            self.shutdown_lidar()

    # === Write live JSON snapshot ===
    def write_json(self, rev, angs, dists):
        try:
            payload = {
                "revolution": int(rev),
                "angle_min_deg": float(np.min(angs)),
                "angle_max_deg": float(np.max(angs)),
                "angle_increment_deg": ANG_RES_DEG,
                "range_min_m": DIST_MIN_MM / 1000.0,
                "range_max_m": DIST_MAX_MM / 1000.0,
                "ranges_m": np.round(dists / 1000.0, 3).tolist(),
                "count": len(dists),
                "timestamp": time.time(),
            }
            JSON_PATH.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            self.get_logger().warn(f"JSON write error: {e}")

    # === Shutdown cleanly ===
    def shutdown_lidar(self):
        try:
            if hasattr(self, "lidar") and self.lidar:
                if getattr(self.lidar, "_serial_port", None) and \
                   self.lidar._serial_port.is_open:
                    self.lidar.stop()
                    self.lidar.stop_motor()
                    self.lidar.disconnect()
            self.get_logger().info("LIDAR stopped cleanly.")
        except Exception as e:
            self.get_logger().warn(f"Shutdown issue: {e}")

    def destroy_node(self):
        self.keep_running = False
        self.shutdown_lidar()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RPLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
