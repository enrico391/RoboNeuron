#!/usr/bin/env python3
"""
control_mcp.py

MCP Server for managing Kinematic Control nodes.
Allows the LLM to bind specific URDFs to action command topics.
"""

import multiprocessing
import re
import tempfile
import time
import numpy as np
import xml.etree.ElementTree as ET
from ikpy.chain import Chain

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from mcp.server.fastmcp import FastMCP

_CONTROL_PROCESS = None
_CONTROL_TOPIC_PROCESS = None
mcp = FastMCP("robomcp-control")

# --- ROS Logic ---

class AutoIKNode(Node):
    """
    ROS2 Node for real-time Inverse Kinematics (IK) calculation.

    Subscribes to:
    - Robot joint state feedback (JointState).
    - End-effector Cartesian delta commands (Float64MultiArray).

    Publishes:
    - Joint trajectory commands (JointTrajectory) OR JointState commands derived from IK solution.
    """
    def __init__(self, urdf_path, cartesian_cmd_topic, state_feedback_topic, joint_cmd_topic, cmd_msg_type="JointTrajectory"):
        super().__init__('auto_ik_node')
        
        self.cmd_msg_type = cmd_msg_type
        self.get_logger().info(f'Subscribing to Cartesian commands on: {cartesian_cmd_topic}')
        self.get_logger().info(f'Subscribing to Joint States on: {state_feedback_topic}')
        self.get_logger().info(f'Publishing {self.cmd_msg_type} to: {joint_cmd_topic}')
        
        # URDF Processing
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        link_parent_map = {}
        link_names = set()
        detected_gripper_joints = []
        for joint in root.findall('joint'):
            name = joint.get('name')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            link_names.add(parent); link_names.add(child)
            link_parent_map[child] = parent
            if joint.get('type') == 'prismatic' or 'finger' in (name.lower() if name else ''):
                detected_gripper_joints.append(name)
        
        base_link_name = list(link_names - set(link_parent_map.keys()))[0]
        
        with open(urdf_path, 'r') as f:
            xml_str = re.sub(r'<visual>.*?</visual>', '', f.read(), flags=re.DOTALL)
            xml_str = re.sub(r'<collision>.*?</collision>', '', xml_str, flags=re.DOTALL)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
            tmp.write(xml_str)
            clean_urdf_path = tmp.name

        self.chain = Chain.from_urdf_file(clean_urdf_path, base_elements=[base_link_name])
        self.gripper_joints = detected_gripper_joints
        self.current_joints = {}
        
        # Determine active joints and extract bounds
        self.ik_mask = []
        self.active_joint_names = []
        for link in self.chain.links:
            if link.joint_type == 'fixed' or link.name == base_link_name:
                self.ik_mask.append(False)
            else:
                self.ik_mask.append(True)
                self.active_joint_names.append(link.name)

        self.create_subscription(JointState, state_feedback_topic, self.state_cb, 10)
        self.create_subscription(Float64MultiArray, cartesian_cmd_topic, self.cmd_cb, 10)
        
        # Conditional Publisher creation based on message type
        if self.cmd_msg_type == "JointState":
            self.pub_cmd = self.create_publisher(JointState, joint_cmd_topic, 10)
        else:
            self.pub_cmd = self.create_publisher(JointTrajectory, joint_cmd_topic, 10)

    def state_cb(self, msg):
        """Processes incoming JointState messages and updates the current joint positions."""
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def cmd_cb(self, msg):
        """
        Receives Cartesian command deltas, calculates IK, and publishes the result 
        as a JointTrajectory or JointState message.
        """
        if not self.current_joints: return
        
        # 1. Construct bounded initial position for IK
        current_ik_q = [0.0] * len(self.chain.links)
        for i, link in enumerate(self.chain.links):
            min_limit, max_limit = link.bounds
            q = 0.0 # Default fallback
            
            if link.name in self.current_joints: 
                q = self.current_joints[link.name]

            # Boundary clipping to ensure valid initial guess
            if min_limit is not None and max_limit is not None:
                q = np.clip(q, min_limit, max_limit)
                
            current_ik_q[i] = q
            
        # 2. Calculate target pose (FK -> Delta -> Pose)
        current_pose = self.chain.forward_kinematics(current_ik_q)
        dx, dy, dz, dr, dp, dy_aw, grip_cmd = msg.data
        
        # RPY Matrix construction
        cx, sx = np.cos(dr), np.sin(dr)
        cy, sy = np.cos(dp), np.sin(dp)
        cz, sz = np.cos(dy_aw), np.sin(dy_aw)
        R = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ \
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            
        delta_pose = np.eye(4)
        delta_pose[:3, 3] = [dx, dy, dz]
        delta_pose[:3, :3] = R
        target_pose = current_pose @ delta_pose
        
        # 3. Solve IK
        try:
            target_ik_q = self.chain.inverse_kinematics_frame(
                target_pose, 
                initial_position=current_ik_q, 
                orientation_mode='all'
            )
        except Exception as e:
            self.get_logger().error(f"IK failed: {e}")
            return

        # Prepare joint names and positions
        final_joint_names = self.active_joint_names + self.gripper_joints
        
        active_positions = []
        for i, link in enumerate(self.chain.links):
            if i < len(self.ik_mask) and self.ik_mask[i]:
                active_positions.append(target_ik_q[i])
                
        gripper_positions = [grip_cmd * 0.04] * len(self.gripper_joints)
        final_positions = active_positions + gripper_positions

        # 4. Package and Publish message
        if self.cmd_msg_type == "JointState":
            out_msg = JointState()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.name = final_joint_names
            out_msg.position = final_positions
            # Velocity and Effort are left empty
            self.pub_cmd.publish(out_msg)
        else:
            out_msg = JointTrajectory()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.joint_names = final_joint_names
            
            point = JointTrajectoryPoint()
            point.positions = final_positions
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 500000000

            out_msg.points.append(point)
            self.pub_cmd.publish(out_msg)

def _ros_worker(urdf_path: str, cartesian_cmd_topic: str, state_feedback_topic: str, joint_cmd_topic: str, cmd_msg_type: str):
    """Worker function to initialize and run the AutoIKNode in a separate process."""
    rclpy.init()
    node = AutoIKNode(urdf_path, cartesian_cmd_topic, state_feedback_topic, joint_cmd_topic, cmd_msg_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

class AutoIKNodeWithURDFSubscriber(Node):
    """
    ROS2 Node for real-time IK calculation with URDF loaded from topic.
    
    Subscribes to:
    - /robot_description (or custom topic) for URDF XML string
    - Robot joint state feedback (JointState)
    - End-effector Cartesian delta commands (Float64MultiArray)
    
    Publishes:
    - Joint trajectory commands (JointTrajectory) OR JointState commands
    """
    def __init__(self, urdf_topic: str, cartesian_cmd_topic: str, state_feedback_topic: str, 
                 joint_cmd_topic: str, cmd_msg_type: str = "JointTrajectory", timeout: float = 10.0):
        super().__init__('auto_ik_node_with_urdf_sub')
        
        self.cmd_msg_type = cmd_msg_type
        self.urdf_received = False
        self.chain = None
        self.gripper_joints = []
        self.current_joints = {}
        self.ik_mask = []
        self.active_joint_names = []
        
        self.get_logger().info(f'Waiting for URDF on topic: {urdf_topic}')
        
        # QoS profile matching /robot_description publisher
        # TRANSIENT_LOCAL is critical - ensures late subscribers receive the URDF
        urdf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to URDF topic
        self.urdf_sub = self.create_subscription(
            String,
            urdf_topic,
            self.urdf_callback,
            urdf_qos
        )
        
        # Wait for URDF with timeout
        start_time = time.time()
        while not self.urdf_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.urdf_received:
            raise RuntimeError(f"Timeout waiting for URDF on {urdf_topic} after {timeout}s")
        
        self.get_logger().info(f'URDF received, initializing controller...')
        self.get_logger().info(f'Subscribing to Cartesian commands on: {cartesian_cmd_topic}')
        self.get_logger().info(f'Subscribing to Joint States on: {state_feedback_topic}')
        self.get_logger().info(f'Publishing {self.cmd_msg_type} to: {joint_cmd_topic}')
        
        # Create subscriptions for control
        self.create_subscription(JointState, state_feedback_topic, self.state_cb, 10)
        self.create_subscription(Float64MultiArray, cartesian_cmd_topic, self.cmd_cb, 10)
        
        # Conditional Publisher creation
        if self.cmd_msg_type == "JointState":
            self.pub_cmd = self.create_publisher(JointState, joint_cmd_topic, 10)
        else:
            self.pub_cmd = self.create_publisher(JointTrajectory, joint_cmd_topic, 10)
    
    def urdf_callback(self, msg: String):
        """Process URDF XML from topic and initialize IK chain."""
        if self.urdf_received:
            return  # Already initialized
        
        urdf_content = msg.data
        
        if not urdf_content or '<robot' not in urdf_content:
            self.get_logger().warn('Received invalid URDF (no <robot> tag)')
            return
        
        try:
            # Parse URDF XML
            root = ET.fromstring(urdf_content)
            link_parent_map = {}
            link_names = set()
            detected_gripper_joints = []
            
            for joint in root.findall('joint'):
                name = joint.get('name')
                parent_elem = joint.find('parent')
                child_elem = joint.find('child')
                
                if parent_elem is not None and child_elem is not None:
                    parent = parent_elem.get('link')
                    child = child_elem.get('link')
                    link_names.add(parent)
                    link_names.add(child)
                    link_parent_map[child] = parent
                    
                    if joint.get('type') == 'prismatic' or 'finger' in (name.lower() if name else ''):
                        detected_gripper_joints.append(name)
            
            base_link_name = list(link_names - set(link_parent_map.keys()))[0]
            
            # Strip visual and collision elements for ikpy
            xml_str = re.sub(r'<visual>.*?</visual>', '', urdf_content, flags=re.DOTALL)
            xml_str = re.sub(r'<collision>.*?</collision>', '', xml_str, flags=re.DOTALL)
            
            # Save to temp file for ikpy
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
                tmp.write(xml_str)
                clean_urdf_path = tmp.name
            
            self.chain = Chain.from_urdf_file(clean_urdf_path, base_elements=[base_link_name])
            self.gripper_joints = detected_gripper_joints
            
            # Determine active joints
            for link in self.chain.links:
                if link.joint_type == 'fixed' or link.name == base_link_name:
                    self.ik_mask.append(False)
                else:
                    self.ik_mask.append(True)
                    self.active_joint_names.append(link.name)
            
            self.urdf_received = True
            self.get_logger().info('URDF processed successfully, controller ready')
            
        except Exception as e:
            self.get_logger().error(f'Failed to process URDF: {e}')
    
    def state_cb(self, msg):
        """Processes incoming JointState messages and updates the current joint positions."""
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos
    
    def cmd_cb(self, msg):
        """
        Receives Cartesian command deltas, calculates IK, and publishes the result.
        """
        if not self.urdf_received or self.chain is None:
            return
        
        if not self.current_joints:
            return
        
        # 1. Construct bounded initial position for IK
        current_ik_q = [0.0] * len(self.chain.links)
        for i, link in enumerate(self.chain.links):
            min_limit, max_limit = link.bounds
            q = 0.0
            
            if link.name in self.current_joints:
                q = self.current_joints[link.name]
            
            if min_limit is not None and max_limit is not None:
                q = np.clip(q, min_limit, max_limit)
            
            current_ik_q[i] = q
        
        # 2. Calculate target pose
        current_pose = self.chain.forward_kinematics(current_ik_q)
        dx, dy, dz, dr, dp, dy_aw, grip_cmd = msg.data
        
        # RPY Matrix construction
        cx, sx = np.cos(dr), np.sin(dr)
        cy, sy = np.cos(dp), np.sin(dp)
        cz, sz = np.cos(dy_aw), np.sin(dy_aw)
        R = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ \
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        
        delta_pose = np.eye(4)
        delta_pose[:3, 3] = [dx, dy, dz]
        delta_pose[:3, :3] = R
        target_pose = current_pose @ delta_pose
        
        # 3. Solve IK
        try:
            target_ik_q = self.chain.inverse_kinematics_frame(
                target_pose,
                initial_position=current_ik_q,
                orientation_mode='all'
            )
        except Exception as e:
            self.get_logger().error(f"IK failed: {e}")
            return
        
        # Prepare joint names and positions
        final_joint_names = self.active_joint_names + self.gripper_joints
        
        active_positions = []
        for i, link in enumerate(self.chain.links):
            if i < len(self.ik_mask) and self.ik_mask[i]:
                active_positions.append(target_ik_q[i])
        
        gripper_positions = [grip_cmd * 0.04] * len(self.gripper_joints)
        final_positions = active_positions + gripper_positions
        
        # 4. Package and Publish message
        if self.cmd_msg_type == "JointState":
            out_msg = JointState()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.name = final_joint_names
            out_msg.position = final_positions
            self.pub_cmd.publish(out_msg)
        else:
            out_msg = JointTrajectory()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.joint_names = final_joint_names
            
            point = JointTrajectoryPoint()
            point.positions = final_positions
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 500000000
            
            out_msg.points.append(point)
            self.pub_cmd.publish(out_msg)

def _ros_worker_with_urdf_topic(urdf_topic: str, cartesian_cmd_topic: str, state_feedback_topic: str, 
                                 joint_cmd_topic: str, cmd_msg_type: str, timeout: float):
    """Worker function to run AutoIKNodeWithURDFSubscriber in a separate process."""
    rclpy.init()
    node = None
    try:
        node = AutoIKNodeWithURDFSubscriber(urdf_topic, cartesian_cmd_topic, state_feedback_topic, 
                                            joint_cmd_topic, cmd_msg_type, timeout)
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# --- MCP Tools ---
@mcp.tool()
def start_controller(urdf_path: str, 
                     cartesian_cmd_topic: str = "/ee_command", 
                     state_feedback_topic: str = "/isaac_joint_states", 
                     joint_cmd_topic: str = "/isaac_joint_commands",
                     cmd_msg_type: str = "JointTrajectory") -> str:
    """
    [ACTION/CONTROL] Starts the Inverse Kinematics (IK) controller loop.
    
    Role in VLA Task: This must be running before starting the VLA inference node, 
    as it receives the predicted actions from the VLA model (via cartesian_cmd_topic) 
    and drives the robot's joints.
    
    Args:
        urdf_path: Path to the robot's URDF file (e.g., for the Panda arm).
        cartesian_cmd_topic: [Input Topic] The topic where the IK controller listens for EE commands (default: /ee_command).
        state_feedback_topic: The topic providing the robot's current joint state feedback (default: /isaac_joint_states).
        joint_cmd_topic: The topic used to publish joint trajectory commands to the robot (default: /isaac_joint_commands).
        cmd_msg_type: The message type used for the output commands ("JointTrajectory" or "JointState").
    """
    global _CONTROL_PROCESS
    if _CONTROL_PROCESS is not None and _CONTROL_PROCESS.is_alive():
        return "Error: Controller is already running."

    if cmd_msg_type not in ["JointTrajectory", "JointState"]:
        return "Error: cmd_msg_type must be 'JointTrajectory' or 'JointState'."

    # Quick validation: check URDF file exists and looks like a robot
    try:
        with open(urdf_path, 'r') as f:
            txt = f.read()
            if '<robot' not in txt:
                return f"Error: file '{urdf_path}' doesn't look like a URDF (no <robot> tag)."
    except Exception as e:
        return f"Error: cannot read urdf_path '{urdf_path}': {e}"

    # Use 'spawn' to avoid fork-related rclpy issues
    ctx = multiprocessing.get_context('spawn')
    _CONTROL_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(urdf_path, cartesian_cmd_topic, state_feedback_topic, joint_cmd_topic, cmd_msg_type),
        daemon=False
    )
    _CONTROL_PROCESS.start()
    return f"Success: Controller started with {urdf_path} (pid={_CONTROL_PROCESS.pid}, type={cmd_msg_type})."

@mcp.tool()
def stop_controller() -> str:
    """Stops the running IK controller process."""
    global _CONTROL_PROCESS
    if _CONTROL_PROCESS is None or not _CONTROL_PROCESS.is_alive():
        return "Info: No controller is running."
    
    _CONTROL_PROCESS.terminate()
    _CONTROL_PROCESS.join(timeout=5.0)
    if _CONTROL_PROCESS.is_alive():
        try:
            _CONTROL_PROCESS.kill()
        except Exception:
            pass
        _CONTROL_PROCESS.join(timeout=1.0)
    _CONTROL_PROCESS = None
    return "Success: Controller stopped."

@mcp.tool()
def start_controller_from_topic(urdf_topic: str = "/robot_description",
                                cartesian_cmd_topic: str = "/ee_command",
                                state_feedback_topic: str = "/isaac_joint_states",
                                joint_cmd_topic: str = "/isaac_joint_commands",
                                cmd_msg_type: str = "JointTrajectory",
                                timeout: float = 10.0) -> str:
    """
    [ACTION/CONTROL] Starts the IK controller by subscribing to robot URDF from a ROS topic.
    
    This is an alternative to start_controller() that doesn't require a URDF file path.
    Instead, it subscribes to a ROS topic (typically /robot_description) where simulators
    or robot drivers publish the robot's URDF description.
    
    Role in VLA Task: Same as start_controller() - must run before VLA inference node.
    Advantage: Works seamlessly with simulators that auto-publish URDF to /robot_description.
    
    Args:
        urdf_topic: Topic publishing URDF as std_msgs/String (default: /robot_description).
        cartesian_cmd_topic: [Input Topic] Where IK controller listens for EE commands (default: /ee_command).
        state_feedback_topic: Topic providing robot joint state feedback (default: /isaac_joint_states).
        joint_cmd_topic: Topic for publishing joint commands to robot (default: /isaac_joint_commands).
        cmd_msg_type: Output message type - "JointTrajectory" or "JointState" (default: "JointTrajectory").
        timeout: Seconds to wait for URDF from topic before failing (default: 10.0).
    """
    global _CONTROL_TOPIC_PROCESS
    if _CONTROL_TOPIC_PROCESS is not None and _CONTROL_TOPIC_PROCESS.is_alive():
        return "Error: Controller from topic is already running."
    
    if cmd_msg_type not in ["JointTrajectory", "JointState"]:
        return "Error: cmd_msg_type must be 'JointTrajectory' or 'JointState'."
    
    ctx = multiprocessing.get_context('spawn')
    _CONTROL_TOPIC_PROCESS = ctx.Process(
        target=_ros_worker_with_urdf_topic,
        args=(urdf_topic, cartesian_cmd_topic, state_feedback_topic, joint_cmd_topic, cmd_msg_type, timeout),
        daemon=False
    )
    _CONTROL_TOPIC_PROCESS.start()
    return f"Success: Controller started from topic {urdf_topic} (pid={_CONTROL_TOPIC_PROCESS.pid}, type={cmd_msg_type})."

@mcp.tool()
def stop_controller_from_topic() -> str:
    """Stops the IK controller that was started from a URDF topic."""
    global _CONTROL_TOPIC_PROCESS
    if _CONTROL_TOPIC_PROCESS is None or not _CONTROL_TOPIC_PROCESS.is_alive():
        return "Info: No topic-based controller is running."
    
    _CONTROL_TOPIC_PROCESS.terminate()
    _CONTROL_TOPIC_PROCESS.join(timeout=5.0)
    if _CONTROL_TOPIC_PROCESS.is_alive():
        try:
            _CONTROL_TOPIC_PROCESS.kill()
        except Exception:
            pass
        _CONTROL_TOPIC_PROCESS.join(timeout=1.0)
    _CONTROL_TOPIC_PROCESS = None
    return "Success: Topic-based controller stopped."

if __name__ == "__main__":
    import argparse
    import time
    import sys
    import select 

    parser = argparse.ArgumentParser(description="control_mcp.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local start/stop test instead of MCP server")
    parser.add_argument("--urdf", type=str, default="../urdf/panda.urdf", help="URDF path to test")
    parser.add_argument("--cartesian-cmd-topic", type=str, default="/ee_command", help="Topic for Cartesian commands (Float64MultiArray)")
    parser.add_argument("--state-feedback-topic", type=str, default="/isaac_joint_states", help="Topic for robot joint state feedback (JointState)")
    parser.add_argument("--joint-cmd-topic", type=str, default="/isaac_joint_commands", help="Topic for publishing joint commands")
    parser.add_argument("--cmd-msg-type", type=str, default="JointState", choices=["JointTrajectory", "JointState"], help="Output message type")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: attempting to start controller (spawn)...")
        res = start_controller(args.urdf, args.cartesian_cmd_topic, args.state_feedback_topic, args.joint_cmd_topic, args.cmd_msg_type)
        print(res)
        if res.startswith("Error"):
            sys.exit(1)

        try:
            print("Controller started. Press Ctrl-C to stop, or type 'stop' + Enter.")
            while True:
                time.sleep(0.5)
                # check child liveness
                if _CONTROL_PROCESS is None or not _CONTROL_PROCESS.is_alive():
                    print("Controller process exited.")
                    break
                # simple stdin check
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() in ("stop", "q", "quit", "exit"):
                        print("Stop command received.")
                        break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, stopping controller...")
        finally:
            print(stop_controller())
            print("Local test finished.")
    else:
        mcp.run()