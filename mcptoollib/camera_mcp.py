#!/usr/bin/env python3
"""
camera_mcp.py

MCP Server for capturing images from a ROS 2 CompressedImage topic.

Subscribes to a sensor_msgs/msg/CompressedImage topic and exposes the latest
frame to LLM agents as a base64-encoded string.  The ROS node is initialized
once at server startup and kept alive for the entire server lifetime.  On each
tool call a short spin yields to the callback thread so the returned image is
as fresh as possible.
"""

import base64
import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("camera-mcp")

_DEFAULT_TOPIC = "/camera0/camera/color/image_raw/compressed"

# ---------------------------------------------------------------------------
# ROS 2 subscriber node (lives for the entire server lifetime)
# ---------------------------------------------------------------------------


class CameraNode(Node):
    """
    Subscribes to a sensor_msgs/CompressedImage topic and caches the latest
    message.  Uses BEST_EFFORT / VOLATILE QoS (standard for camera streams).
    """

    def __init__(self, topic: str):
        super().__init__("mcp_camera_subscriber")
        self._lock = threading.Lock()
        self._latest: CompressedImage | None = None
        self._topic = topic
        self._msg_count: int = 0
        self._last_recv_time: float | None = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(
            CompressedImage, topic, self._callback, qos
        )
        self.get_logger().info(f"Subscribed to {topic}")

    def _callback(self, msg: CompressedImage) -> None:
        with self._lock:
            self._latest = msg
            self._msg_count += 1
            self._last_recv_time = time.time()
        self.get_logger().debug(
            f"Received image from {self._topic} "
            f"(format={msg.format}, size={len(msg.data)} bytes)"
        )

    def get_latest(self) -> CompressedImage | None:
        with self._lock:
            return self._latest

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "topic": self._topic,
                "messages_received": self._msg_count,
                "has_image": self._latest is not None,
                "last_received": self._last_recv_time,
            }


# ---------------------------------------------------------------------------
# Global node + spin thread (initialized once at import time)
# ---------------------------------------------------------------------------

_ros_node: CameraNode | None = None
_spin_thread: threading.Thread | None = None


def _init_ros(topic: str = _DEFAULT_TOPIC) -> None:
    """Initialize rclpy and the subscriber node, then spin in a daemon thread."""
    global _ros_node, _spin_thread

    if _ros_node is not None:
        return  # already initialized

    if not rclpy.ok():
        rclpy.init()

    _ros_node = CameraNode(topic)

    _spin_thread = threading.Thread(
        target=rclpy.spin, args=(_ros_node,), daemon=True
    )
    _spin_thread.start()


# Initialize immediately when the module is imported (server startup).
_init_ros()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_camera_image(topic: str = _DEFAULT_TOPIC) -> str:
    """
    [PERCEPTION/INFO] Capture the latest camera image as a base64-encoded string.

    Subscribes to the given ROS 2 topic (sensor_msgs/CompressedImage) and
    returns the most recent frame.  The response includes:
      - format     : compression format reported by the publisher (e.g. "jpeg")
      - base64     : the image data encoded as a base64 string
      - size_bytes : original compressed size in bytes
      - topic      : the source topic

    The underlying ROS node remains alive for the server lifetime.
    Every call performs a short spin to capture any freshly published frame.

    Args:
        topic: The ROS 2 topic publishing sensor_msgs/CompressedImage.
    """
    global _ros_node

    if _ros_node is None or _ros_node._topic != topic:
        _init_ros(topic)

    # Yield to the spin thread so any pending callback can deliver a frame.
    time.sleep(0.15)

    msg = _ros_node.get_latest()

    if msg is None:
        return json.dumps(
            {
                "error": (
                    f"No image received yet on '{topic}'. "
                    "Ensure the camera is publishing and the topic is active."
                ),
            },
            indent=2,
        )

    img_b64 = base64.b64encode(msg.data).decode("ascii")

    return json.dumps(
        {
            "topic": topic,
            "format": msg.format,
            "size_bytes": len(msg.data),
            "base64": img_b64,
        },
        indent=2,
    )


@mcp.tool()
def camera_status(topic: str = _DEFAULT_TOPIC) -> str:
    """
    [PERCEPTION/INFO] Check the status of the camera image subscriber.

    Returns diagnostic information about the camera subscriber:
      - topic             : the subscribed topic
      - messages_received : total number of images received since startup
      - has_image         : whether at least one image has been received
      - last_received     : timestamp of the last received image (epoch seconds)
      - image_format      : compression format of the last image (if available)
      - image_size_bytes  : size of the last compressed image (if available)

    Args:
        topic: The ROS 2 topic publishing sensor_msgs/CompressedImage.
    """
    global _ros_node

    if _ros_node is None or _ros_node._topic != topic:
        _init_ros(topic)

    # Brief yield so stats are up to date.
    time.sleep(0.05)

    stats = _ros_node.get_stats()

    msg = _ros_node.get_latest()
    if msg is not None:
        stats["image_format"] = msg.format
        stats["image_size_bytes"] = len(msg.data)

    return json.dumps(stats, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Camera MCP Server...")
    mcp.run(transport="stdio")
