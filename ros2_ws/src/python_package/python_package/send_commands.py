import json
from urllib.parse import quote

import requests
import rclpy
from rclpy.node import Node


class SendCommands(Node):
    def __init__(self):
        super().__init__("send_commands")

        # Rover network target
        self.declare_parameter("ip", "192.168.4.1")
        self.ip = str(self.get_parameter("ip").value)

        # How often to send (seconds)
        self.declare_parameter("period_s", 0.2)  # 5 Hz default
        self.period_s = float(self.get_parameter("period_s").value)

        # Simple velocity command (edit keys to match your rover firmware!)
        self.declare_parameter("linear", 0.2)    # forward speed
        self.declare_parameter("angular", 0.0)   # turn rate
        self.linear = float(self.get_parameter("linear").value)
        self.angular = float(self.get_parameter("angular").value)

        # Optional: choose the command "T" code your rover expects
        self.declare_parameter("T", 200)  # CHANGE THIS to your rover's movement command ID
        self.T = int(self.get_parameter("T").value)

        self.session = requests.Session()
        self.timer = self.create_timer(self.period_s, self.send_once)

        self.get_logger().info(
            f"SendCommands started. Target={self.ip}, period={self.period_s}s, "
            f"T={self.T}, linear={self.linear}, angular={self.angular}"
        )

    def send_once(self):
        # IMPORTANT: adjust this payload to EXACTLY match your rover's expected JSON schema.
        command = {
            "T": self.T,
            "v": self.linear,
            "w": self.angular,
        }

        json_str = json.dumps(command, separators=(",", ":"))
        url = f"http://{self.ip}/js?json={quote(json_str)}"

        try:
            response = self.session.get(url, timeout=0.5)
            response.raise_for_status()

            text = response.text.strip()
            # Some firmwares return JSON, some return plain text
            try:
                data = json.loads(text)
                self.get_logger().info(f"Command ack: {data}")
            except Exception:
                # Don’t spam logs too hard at high rate
                self.get_logger().debug(f"Command ack (raw): {text}")

        except Exception as e:
            self.get_logger().warning(f"Failed to send command: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SendCommands()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
