#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    def __init__(self):
        super().__init__("listener")
        self.sub = self.create_subscription(String, "chatter", self.cb, 10)
        self.get_logger().info("Listener started: subscribed to /chatter")

    def cb(self, msg: String):
        self.get_logger().info(f"Received: {msg.data}")


def main():
    rclpy.init()
    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
