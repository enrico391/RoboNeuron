#!/usr/bin/env python3
"""
saved_poses_mcp.py

MCP Server for retrieving saved robot poses from the /saved_poses ROS 2 topic.
The topic type is nav_msgs/msg/Path, where each PoseStamped uses header.frame_id
as the human-readable name of the pose (e.g. "Kitchen", "Bedroom", "Livingroom").

The ROS node is initialized once at server startup and kept alive for the entire
server lifetime. On each tool call a short spin flushes any new message so the
returned data is always up to date.
"""

import json
import threading
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from nav_msgs.msg import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("saved-poses-mcp")

# ---------------------------------------------------------------------------
# ROS 2 subscriber node (lives for the entire server lifetime)
# ---------------------------------------------------------------------------

class SavedPosesNode(Node):
    """
    Subscribes to a nav_msgs/Path topic and caches the latest message.

    Uses TRANSIENT_LOCAL / RELIABLE QoS so it receives the last retained
    message immediately upon subscribing (if the publisher uses the same QoS).
    Falls back gracefully when the publisher uses a different QoS.
    """

    def __init__(self, topic: str):
        super().__init__("mcp_saved_poses_subscriber")
        self._lock = threading.Lock()
        self._latest: Path | None = None
        self._topic = topic

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(Path, topic, self._callback, qos)
        self.get_logger().info(f"Subscribed to {topic} (QoS: RELIABLE / TRANSIENT_LOCAL)")

    def _callback(self, msg: Path) -> None:
        with self._lock:
            self._latest = msg
        self.get_logger().debug(f"Received {len(msg.poses)} poses from {self._topic}")

    def get_latest(self) -> Path | None:
        with self._lock:
            return self._latest


# ---------------------------------------------------------------------------
# Global node + spin thread (initialized once at import time)
# ---------------------------------------------------------------------------

_ros_node: SavedPosesNode | None = None
_spin_thread: threading.Thread | None = None


def _init_ros(topic: str = "/saved_poses") -> None:
    """Initialize rclpy and the subscriber node, then spin in a daemon thread."""
    global _ros_node, _spin_thread

    if _ros_node is not None:
        return  # already initialized

    if not rclpy.ok():
        rclpy.init()

    _ros_node = SavedPosesNode(topic)

    _spin_thread = threading.Thread(target=rclpy.spin, args=(_ros_node,), daemon=True)
    _spin_thread.start()


# Initialize immediately when the module is imported (server startup).
_init_ros()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_to_list(msg: Path) -> list[dict[str, Any]]:
    """Convert a nav_msgs/Path to a plain list of dicts, one per pose."""
    result = []
    for ps in msg.poses:
        result.append(
            {
                "name": ps.header.frame_id,
                "position": {
                    "x": ps.pose.position.x,
                    "y": ps.pose.position.y,
                    "z": ps.pose.position.z,
                },
                "orientation": {
                    "x": ps.pose.orientation.x,
                    "y": ps.pose.orientation.y,
                    "z": ps.pose.orientation.z,
                    "w": ps.pose.orientation.w,
                },
            }
        )
    return result


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_saved_poses(topic: str = "/saved_poses") -> str:
    """
    [NAVIGATION/INFO] Retrieve the list of saved robot poses.

    Subscribes to the given ROS 2 topic (nav_msgs/Path) and returns every
    named pose as a JSON array.  Each entry contains:
      - name       : human-readable label (header.frame_id of each PoseStamped)
      - position   : {x, y, z}
      - orientation: {x, y, z, w}  (quaternion)

    The underlying ROS node remains alive for the server lifetime.
    Every call performs a short spin to capture any new message published
    since the last call, so the list is always current.

    Args:
        topic: The ROS 2 topic publishing nav_msgs/Path (default: /saved_poses).
    """
    global _ros_node

    # If the caller changed the topic compared to the running node, reinitialize.
    if _ros_node is None or _ros_node._topic != topic:
        _init_ros(topic)

    # Give ROS a brief window to deliver any freshly published message.
    # The daemon spin thread is already running; we just yield the GIL for
    # a short moment so its callbacks can execute.
    time.sleep(0.15)

    msg = _ros_node.get_latest()

    if msg is None:
        return json.dumps(
            {
                "error": (
                    f"No message received yet on '{topic}'. "
                    "Ensure the publisher is running and the topic is active."
                ),
                "poses": [],
            },
            indent=2,
        )

    poses = _path_to_list(msg)
    return json.dumps(
        {
            "topic": topic,
            "frame_id": msg.header.frame_id,
            "timestamp": {
                "sec": msg.header.stamp.sec,
                "nanosec": msg.header.stamp.nanosec,
            },
            "count": len(poses),
            "poses": poses,
        },
        indent=2,
    )


@mcp.tool()
def get_pose_by_name(name: str, topic: str = "/saved_poses") -> str:
    """
    [NAVIGATION/INFO] Retrieve a single saved pose by its name.

    Looks up a pose whose header.frame_id matches the given name (case-sensitive).
    Useful before calling navigate_to_pose so you can pass exact coordinates.

    Args:
        name : The pose name to look up (e.g. "Kitchen", "Bedroom").
        topic: The ROS 2 topic publishing nav_msgs/Path (default: /saved_poses).
    """
    global _ros_node

    if _ros_node is None or _ros_node._topic != topic:
        _init_ros(topic)

    time.sleep(0.15)

    msg = _ros_node.get_latest()

    if msg is None:
        return json.dumps(
            {
                "error": (
                    f"No message received yet on '{topic}'. "
                    "Ensure the publisher is running and the topic is active."
                )
            },
            indent=2,
        )

    poses = _path_to_list(msg)
    matches = [p for p in poses if p["name"] == name]

    if not matches:
        available = [p["name"] for p in poses]
        return json.dumps(
            {
                "error": f"Pose '{name}' not found.",
                "available_poses": available,
            },
            indent=2,
        )

    return json.dumps(matches[0], indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Saved Poses MCP Server...")
    mcp.run(transport="stdio")
