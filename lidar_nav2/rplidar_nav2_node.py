#!/usr/bin/env python3
"""
ROS2 Node: RPLidar → /scan publisher (Nav2 compatible, ROS2 Humble)
Converts the lidar stream into sensor_msgs/LaserScan messages.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from rplidar import RPLidar
import numpy as np
import math
from collections import defaultdict

# --- Config ---
PORT = '/dev/ttyUSB0'
BAUD = 115200
ANG_RES_DEG = 1.0
REV_AVG = 2
DIST_MIN_MM = 120
DIST_MAX_MM = 3500
FRAME_ID = 'lidar_link'

def deg_to_idx(deg):
    return int(round(deg % 360 / ANG_RES_DEG))

def robust_bin_average(frames_by_idx):
    idxs, dists = [], []
    for idx, vals in frames_by_idx.items():
        if vals:
            arr = np.array(vals)
            idxs.append(idx)
            dists.append(np.median(arr))
    return np.array(idxs) * ANG_RES_DEG, np.array(dists)

class RPLidarNode(Node):
    def __init__(self):
        super().__init__('rplidar_nav2_node')
        self.publisher_ = self.create_publisher(LaserScan, '/scan', 10)
        self.lidar = RPLidar(PORT, baudrate=BAUD)
        self.get_logger().info(f"Connected: {self.lidar.get_info()}")
        self.get_logger().info(f"Health: {self.lidar.get_health()}")
        self.ring = []
        self.create_timer(0.2, self.publish_scan)  # ~5 Hz

    def publish_scan(self):
        try:
            scan = next(self.lidar.iter_scans(max_buf_meas=2000, min_len=50))
            bins = defaultdict(list)
            for q, angle, dist in scan:
                if DIST_MIN_MM <= dist <= DIST_MAX_MM:
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
                return

            # Build LaserScan message
            msg = LaserScan()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = FRAME_ID

            msg.angle_min = 0.0
            msg.angle_max = 2 * math.pi
            msg.angle_increment = math.radians(ANG_RES_DEG)
            msg.time_increment = 0.0
            msg.scan_time = 0.2
            msg.range_min = DIST_MIN_MM / 1000.0
            msg.range_max = DIST_MAX_MM / 1000.0
            msg.ranges = (dists / 1000.0).tolist()
            msg.intensities = [1.0] * len(dists)

            self.publisher_.publish(msg)
            self.get_logger().info(f"Published scan ({len(dists)} pts)")
        except StopIteration:
            pass
        except Exception as e:
            self.get_logger().error(str(e))

    def destroy_node(self):
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RPLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down…')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
