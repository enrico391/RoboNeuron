#!/usr/bin/env python3
"""
urdf_mcp.py

MCP Server for managing robot URDF descriptions from ROS topics.
Subscribes to /robot_description topic and provides URDF access to other services.
"""

import multiprocessing
import os
import tempfile
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from mcp.server.fastmcp import FastMCP
import sys
import argparse
import time
import select

# Global storage for the background process
_URDF_PROCESS = None
# Shared memory for URDF content
_URDF_CACHE = multiprocessing.Manager().dict() if multiprocessing.current_process().name == 'MainProcess' else {}

mcp = FastMCP("robomcp-urdf")

# --- ROS Logic ---

class URDFSubscriberNode(Node):
    """
    ROS2 Node that subscribes to /robot_description topic and caches URDF content.
    Uses TRANSIENT_LOCAL durability to receive URDF from late-joining subscribers.
    """
    def __init__(self, topic: str, cache_dict: dict):
        super().__init__('urdf_subscriber_node')
        self.cache = cache_dict
        
        # QoS profile matching /robot_description publisher
        # TRANSIENT_LOCAL is critical - ensures late subscribers receive the URDF
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            String,
            topic,
            self.urdf_callback,
            qos_profile
        )
        self.get_logger().info(f'Subscribed to {topic} for URDF updates (QoS: RELIABLE, TRANSIENT_LOCAL)')
        
    def urdf_callback(self, msg: String):
        """
        Callback when URDF is received from the topic.
        Stores the URDF XML string in shared cache.
        """
        urdf_content = msg.data
        if urdf_content and '<robot' in urdf_content:
            self.cache['urdf'] = urdf_content
            self.cache['timestamp'] = time.time()
            self.get_logger().info(f'URDF received and cached ({len(urdf_content)} bytes)')
        else:
            self.get_logger().warn('Received invalid URDF (no <robot> tag)')

def _ros_worker(topic: str, cache_dict: dict):
    """Worker function to run the URDF subscriber node in a separate process."""
    rclpy.init()
    node = URDFSubscriberNode(topic, cache_dict)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# --- MCP Tools ---

@mcp.tool()
def start_urdf_subscriber(topic: str = "/robot_description") -> str:
    """
    [URDF/INFO] Starts subscribing to the robot URDF description topic.
    
    The URDF (Unified Robot Description Format) describes the robot's kinematic structure.
    This tool subscribes to a ROS topic where simulators or robot drivers publish the URDF,
    making it available for kinematic calculations without needing file paths.
    
    Args:
        topic: The ROS topic publishing the URDF as a String message (default: /robot_description).
    """
    global _URDF_PROCESS, _URDF_CACHE
    
    if _URDF_PROCESS is not None and _URDF_PROCESS.is_alive():
        return "Error: URDF subscriber is already running."
    
    # Initialize cache manager if needed
    if not isinstance(_URDF_CACHE, dict) or not hasattr(_URDF_CACHE, '_manager'):
        _URDF_CACHE = multiprocessing.Manager().dict()
    
    # Use 'spawn' to avoid fork-related rclpy issues
    ctx = multiprocessing.get_context('spawn')
    _URDF_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(topic, _URDF_CACHE),
        daemon=False
    )
    _URDF_PROCESS.start()
    return f"Success: URDF subscriber started on {topic} (pid={_URDF_PROCESS.pid})."

@mcp.tool()
def stop_urdf_subscriber() -> str:
    """Stops the URDF subscriber node."""
    global _URDF_PROCESS
    if _URDF_PROCESS is None or not _URDF_PROCESS.is_alive():
        return "Info: No URDF subscriber is running."
    
    _URDF_PROCESS.terminate()
    _URDF_PROCESS.join(timeout=5.0)
    if _URDF_PROCESS.is_alive():
        try:
            _URDF_PROCESS.kill()
        except Exception:
            pass
        _URDF_PROCESS.join(timeout=1.0)
    _URDF_PROCESS = None
    return "Success: URDF subscriber stopped."

@mcp.tool()
def get_urdf() -> str:
    """
    [URDF/INFO] Retrieves the cached URDF content.
    
    Returns the latest URDF XML string received from the /robot_description topic.
    If no URDF has been received yet, returns an error message.
    
    Returns:
        The URDF XML content as a string, or an error message if unavailable.
    """
    global _URDF_CACHE
    
    if 'urdf' not in _URDF_CACHE:
        return "Error: No URDF available. Ensure start_urdf_subscriber() has been called and the topic is publishing."
    
    return _URDF_CACHE['urdf']

@mcp.tool()
def save_urdf_to_file(filepath: str) -> str:
    """
    [URDF/ACTION] Saves the cached URDF to a file.
    
    Writes the currently cached URDF XML content to the specified file path.
    Useful when other tools require a file path instead of string content.
    
    Args:
        filepath: Destination path for the URDF file (e.g., "/tmp/robot.urdf").
    
    Returns:
        Success message with file path, or error if URDF is unavailable.
    """
    global _URDF_CACHE
    
    if 'urdf' not in _URDF_CACHE:
        return "Error: No URDF available. Call start_urdf_subscriber() first."
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(_URDF_CACHE['urdf'])
        
        return f"Success: URDF saved to {filepath} ({len(_URDF_CACHE['urdf'])} bytes)."
    except Exception as e:
        return f"Error: Failed to save URDF to {filepath}: {e}"

@mcp.tool()
def get_temp_urdf_path() -> str:
    """
    [URDF/ACTION] Saves the cached URDF to a temporary file and returns the path.
    
    Creates a temporary .urdf file with the cached URDF content. The file persists
    until manually deleted or system cleanup. Useful for tools that require file paths.
    
    Returns:
        Temporary file path containing the URDF, or error if unavailable.
    """
    global _URDF_CACHE
    
    if 'urdf' not in _URDF_CACHE:
        return "Error: No URDF available. Call start_urdf_subscriber() first."
    
    try:
        # Create a named temporary file that persists after closing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
            tmp.write(_URDF_CACHE['urdf'])
            tmp_path = tmp.name
        
        return f"Success: {tmp_path}"
    except Exception as e:
        return f"Error: Failed to create temporary URDF file: {e}"

@mcp.tool()
def urdf_status() -> str:
    """
    [URDF/INFO] Checks the status of the URDF subscriber and cached data.
    
    Returns information about whether the subscriber is running, if URDF data
    has been received, and when it was last updated.
    """
    global _URDF_PROCESS, _URDF_CACHE
    
    status_lines = []
    
    # Check process status
    if _URDF_PROCESS is None:
        status_lines.append("Subscriber Status: Not started")
    elif _URDF_PROCESS.is_alive():
        status_lines.append(f"Subscriber Status: Running (pid={_URDF_PROCESS.pid})")
    else:
        status_lines.append("Subscriber Status: Stopped")
    
    # Check cache status
    if 'urdf' in _URDF_CACHE:
        urdf_size = len(_URDF_CACHE['urdf'])
        timestamp = _URDF_CACHE.get('timestamp', 'unknown')
        status_lines.append(f"URDF Cache: Available ({urdf_size} bytes)")
        status_lines.append(f"Last Updated: {time.ctime(timestamp) if isinstance(timestamp, (int, float)) else timestamp}")
    else:
        status_lines.append("URDF Cache: Empty (waiting for data)")
    
    return "\n".join(status_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="urdf_mcp.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local test instead of MCP server")
    parser.add_argument("--topic", type=str, default="/robot_description", help="Topic to subscribe for URDF")
    args = parser.parse_args()
    
    if args.local_test:
        print("LOCAL TEST MODE: starting URDF subscriber...")
        
        # Initialize shared cache for local test
        _URDF_CACHE = multiprocessing.Manager().dict()
        
        res = start_urdf_subscriber(args.topic)
        print(res)
        if res.startswith("Error"):
            sys.exit(1)
        
        try:
            print(f"URDF subscriber started. Waiting for URDF on {args.topic}...")
            print("Press Ctrl-C to stop, or type 'status' to check, 'get' to display URDF, 'stop' to exit.")
            
            while True:
                time.sleep(0.5)
                
                # Check if process is alive
                if _URDF_PROCESS is None or not _URDF_PROCESS.is_alive():
                    print("Subscriber process exited.")
                    break
                
                # Check for user input
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip().lower()
                    
                    if line in ("stop", "q", "quit", "exit"):
                        print("Stop command received.")
                        break
                    elif line == "status":
                        print("\n" + urdf_status() + "\n")
                    elif line == "get":
                        urdf = get_urdf()
                        if urdf.startswith("Error"):
                            print(urdf)
                        else:
                            print(f"\nURDF Content ({len(urdf)} bytes):")
                            print(urdf[:500] + "..." if len(urdf) > 500 else urdf)
                            print()
                    elif line == "save":
                        test_path = "/tmp/test_robot.urdf"
                        result = save_urdf_to_file(test_path)
                        print(result)
        
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, stopping subscriber...")
        finally:
            print(stop_urdf_subscriber())
            print("Local test finished.")
    else:
        # Normal MCP server mode
        mcp.run()
