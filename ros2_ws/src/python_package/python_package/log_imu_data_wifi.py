import json
from urllib.parse import quote

import requests
import rclpy
from rclpy.node import Node


class ImuLogWifi(Node):
    def __init__(self):
        super().__init__("imu_log_wifi")

        # Declare + read parameter correctly
        self.declare_parameter("ip", "192.168.4.1")
        self.ip = str(self.get_parameter("ip").value)

        # Poll period (seconds). 1.0s = 1 Hz
        self.period_s = 1.0

        self.session = requests.Session()
        self.timer = self.create_timer(self.period_s, self.poll_imu)

        self.get_logger().info(
            f"IMU Wifi Logger started. Polling rover at {self.ip} every {self.period_s:.1f}s"
        )

    def poll_imu(self):
        command = {"T": 126}
        json_str = json.dumps(command, separators=(",", ":"))
        url = f"http://{self.ip}/js?json={quote(json_str)}"

        try:
            response = self.session.get(url, timeout=0.5)
            response.raise_for_status()

            text = response.text.strip()
            try:
                data = json.loads(text)
                self.get_logger().info(f"IMU Data: {data}")
            except Exception:
                self.get_logger().info(f"IMU Raw: {text}")

        except Exception as e:
            self.get_logger().warning(f"Failed to poll IMU: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImuLogWifi()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
