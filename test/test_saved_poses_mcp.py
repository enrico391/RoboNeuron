#!/usr/bin/env python3
"""
test_saved_poses_mcp.py

Unit tests for mcptoollib/saved_poses_mcp.py MCP server.

All ROS 2 dependencies are mocked so the tests run without a live ROS
environment.  The module-level rclpy.init() / spin thread is also bypassed
so the import itself is side-effect-free.

Run with:
    pytest test/test_saved_poses_mcp.py -v
"""

import importlib
import json
import sys
import threading
import time
import types
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ROS / MCP stubs — injected before the module under test is imported
# ---------------------------------------------------------------------------

def _make_stubs():
    """Return a minimal set of stub modules that satisfy saved_poses_mcp imports."""

    # rclpy
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = MagicMock()
    rclpy_mod.ok = MagicMock(return_value=True)
    rclpy_mod.shutdown = MagicMock()
    rclpy_mod.spin = MagicMock()  # called in the daemon thread

    rclpy_node = types.ModuleType("rclpy.node")
    class _NodeBase:
        def __init__(self, *args, **kwargs):
            pass
        def create_subscription(self, *args, **kwargs):
            return MagicMock()
        def get_logger(self):
            return MagicMock()
    rclpy_node.Node = _NodeBase

    rclpy_qos = types.ModuleType("rclpy.qos")
    for cls_name in ("QoSProfile", "ReliabilityPolicy", "DurabilityPolicy", "HistoryPolicy"):
        setattr(rclpy_qos, cls_name, MagicMock())

    # nav_msgs
    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")

    class _Path:
        """Minimal nav_msgs/Path stub."""
        def __init__(self):
            self.header = _Header()
            self.poses: list = []

    class _Header:
        def __init__(self):
            self.frame_id = ""
            self.stamp = _Stamp()

    class _Stamp:
        def __init__(self):
            self.sec = 0
            self.nanosec = 0

    class _PoseStamped:
        def __init__(self):
            self.header = _Header()
            self.pose = _Pose()

    class _Pose:
        def __init__(self):
            self.position = _Point()
            self.orientation = _Quaternion()

    class _Point:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _Quaternion:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    nav_msgs_msg.Path = _Path
    nav_msgs_msg.PoseStamped = _PoseStamped

    # mcp
    mcp_mod = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    mcp_fastmcp = types.ModuleType("mcp.server.fastmcp")

    class _FastMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            """Passthrough decorator – does not wrap the function."""
            def decorator(fn):
                return fn
            return decorator

        def run(self, **kwargs):
            pass

    mcp_fastmcp.FastMCP = _FastMCP

    return {
        "rclpy": rclpy_mod,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "nav_msgs": nav_msgs,
        "nav_msgs.msg": nav_msgs_msg,
        "mcp": mcp_mod,
        "mcp.server": mcp_server,
        "mcp.server.fastmcp": mcp_fastmcp,
    }


# ---------------------------------------------------------------------------
# Fixture: import module under test with stubs active
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def saved_poses_mcp():
    """
    Import (or reload) saved_poses_mcp with all ROS stubs active and the
    module-level _init_ros() call patched to a no-op.
    """
    stubs = _make_stubs()

    # Patch threading.Thread so the daemon spin thread is never actually started
    # during import.  We replace it with a no-op Thread that does not .start().
    dummy_thread = MagicMock(spec=threading.Thread)

    with mock.patch.dict(sys.modules, stubs), \
         mock.patch("threading.Thread", return_value=dummy_thread):
        import mcptoollib.saved_poses_mcp as mod
        importlib.reload(mod)
        return mod


@pytest.fixture(autouse=True, scope="module")
def patch_ros_imports():
    """Keep ROS stubs injected for the entire test module (mirrors test_urdf_mcp.py)."""
    stubs = _make_stubs()
    with mock.patch.dict(sys.modules, stubs):
        yield

# ---------------------------------------------------------------------------
# Helpers: build fake nav_msgs/Path messages
# ---------------------------------------------------------------------------

def _make_pose(name: str, x: float, y: float, z: float = 0.0,
               ox: float = 0.0, oy: float = 0.0, oz: float = 0.0, ow: float = 1.0):
    """Return a minimal PoseStamped-like object."""
    class _H:
        frame_id = name
        class stamp:
            sec = 0
            nanosec = 0

    class _Pos:
        pass

    class _Ori:
        pass

    class _Pose:
        pass

    class _PS:
        pass

    pos = _Pos()
    pos.x, pos.y, pos.z = x, y, z

    ori = _Ori()
    ori.x, ori.y, ori.z, ori.w = ox, oy, oz, ow

    p = _Pose()
    p.position = pos
    p.orientation = ori

    ps = _PS()
    ps.header = _H()
    ps.pose = p
    return ps


def _make_path(frame_id: str = "map", sec: int = 1772368852, poses=None):
    """Return a minimal Path-like object."""
    class _Stamp:
        pass

    class _Header:
        pass

    class _Path:
        pass

    stamp = _Stamp()
    stamp.sec = sec
    stamp.nanosec = 0

    header = _Header()
    header.frame_id = frame_id
    header.stamp = stamp

    path = _Path()
    path.header = header
    path.poses = poses or []
    return path


SAMPLE_PATH = _make_path(poses=[
    _make_pose("Banco",          x=7.45,  y=-2.36, oz=-0.2474, ow=0.9689),
    _make_pose("sala",           x=6.94,  y=-9.21, oz=-0.2426, ow=0.9701),
    _make_pose("inizio camera",  x=4.36,  y=-4.94, oz=-0.9107, ow=0.4130),
])


# ---------------------------------------------------------------------------
# Tests: _init_ros
# ---------------------------------------------------------------------------

class TestInitRos:

    def test_initializes_node_and_thread(self, saved_poses_mcp):
        """_init_ros() should create a SavedPosesNode and start a daemon thread."""
        mock_node = MagicMock()
        mock_thread = MagicMock()
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True

        with patch.object(saved_poses_mcp, "_ros_node", None), \
             patch.object(saved_poses_mcp, "_spin_thread", None), \
             patch.object(saved_poses_mcp, "rclpy", mock_rclpy), \
             patch.object(saved_poses_mcp, "SavedPosesNode", return_value=mock_node) as node_cls, \
             patch("threading.Thread", return_value=mock_thread) as thread_cls:

            saved_poses_mcp._init_ros("/saved_poses")

        node_cls.assert_called_once_with("/saved_poses")
        mock_thread.start.assert_called_once()

    def test_skips_reinit_if_node_already_set(self, saved_poses_mcp):
        """_init_ros() is a no-op when a node is already initialized."""
        existing_node = MagicMock()

        with patch.object(saved_poses_mcp, "_ros_node", existing_node), \
             patch("mcptoollib.saved_poses_mcp.SavedPosesNode") as node_cls:

            saved_poses_mcp._init_ros("/saved_poses")

        node_cls.assert_not_called()

    def test_calls_rclpy_init_when_not_ok(self, saved_poses_mcp):
        """Should call rclpy.init() when rclpy is not yet initialized."""
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = False
        mock_node = MagicMock()

        with patch.object(saved_poses_mcp, "_ros_node", None), \
             patch.object(saved_poses_mcp, "_spin_thread", None), \
             patch.object(saved_poses_mcp, "rclpy", mock_rclpy), \
             patch.object(saved_poses_mcp, "SavedPosesNode", return_value=mock_node), \
             patch("threading.Thread", return_value=MagicMock()):

            saved_poses_mcp._init_ros()

        mock_rclpy.init.assert_called()


# ---------------------------------------------------------------------------
# Tests: _path_to_list
# ---------------------------------------------------------------------------

class TestPathToList:

    def test_converts_all_poses(self, saved_poses_mcp):
        result = saved_poses_mcp._path_to_list(SAMPLE_PATH)
        assert len(result) == 3

    def test_pose_names_match_frame_ids(self, saved_poses_mcp):
        result = saved_poses_mcp._path_to_list(SAMPLE_PATH)
        assert result[0]["name"] == "Banco"
        assert result[1]["name"] == "sala"
        assert result[2]["name"] == "inizio camera"

    def test_position_values_are_correct(self, saved_poses_mcp):
        result = saved_poses_mcp._path_to_list(SAMPLE_PATH)
        banco = result[0]
        assert banco["position"]["x"] == pytest.approx(7.45)
        assert banco["position"]["y"] == pytest.approx(-2.36)
        assert banco["position"]["z"] == pytest.approx(0.0)

    def test_orientation_values_are_correct(self, saved_poses_mcp):
        result = saved_poses_mcp._path_to_list(SAMPLE_PATH)
        banco = result[0]
        assert banco["orientation"]["w"] == pytest.approx(0.9689, abs=1e-3)
        assert banco["orientation"]["z"] == pytest.approx(-0.2474, abs=1e-3)

    def test_empty_path_returns_empty_list(self, saved_poses_mcp):
        result = saved_poses_mcp._path_to_list(_make_path(poses=[]))
        assert result == []

    def test_output_keys_are_complete(self, saved_poses_mcp):
        result = saved_poses_mcp._path_to_list(SAMPLE_PATH)
        for entry in result:
            assert "name" in entry
            assert "position" in entry
            assert "orientation" in entry
            assert set(entry["position"]) == {"x", "y", "z"}
            assert set(entry["orientation"]) == {"x", "y", "z", "w"}


# ---------------------------------------------------------------------------
# Tests: get_saved_poses
# ---------------------------------------------------------------------------

class TestGetSavedPoses:

    def test_returns_all_poses_as_json(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses()

        data = json.loads(raw)
        assert data["count"] == 3
        assert len(data["poses"]) == 3

    def test_pose_names_present_in_output(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses()

        data = json.loads(raw)
        names = [p["name"] for p in data["poses"]]
        assert names == ["Banco", "sala", "inizio camera"]

    def test_returns_topic_and_frame_id_metadata(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses()

        data = json.loads(raw)
        assert data["topic"] == "/saved_poses"
        assert data["frame_id"] == "map"
        assert "timestamp" in data
        assert data["timestamp"]["sec"] == 1772368852

    def test_returns_error_json_when_no_message(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = None

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses()

        data = json.loads(raw)
        assert "error" in data
        assert data["poses"] == []

    def test_error_message_mentions_topic(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = None

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses(topic="/saved_poses")

        data = json.loads(raw)
        assert "/saved_poses" in data["error"]

    def test_reinitializes_node_on_topic_change(self, saved_poses_mcp):
        """When a different topic is requested the node should be re-initialized."""
        old_node = MagicMock()
        old_node._topic = "/saved_poses"
        old_node.get_latest.return_value = None

        with patch.object(saved_poses_mcp, "_ros_node", old_node), \
             patch.object(saved_poses_mcp, "_init_ros") as mock_init, \
             patch("time.sleep"):
            saved_poses_mcp.get_saved_poses(topic="/other_poses")

        mock_init.assert_called_once_with("/other_poses")

    def test_output_is_valid_json(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses()

        # Must not raise
        json.loads(raw)

    def test_empty_path_returns_count_zero(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = _make_path(poses=[])

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_saved_poses()

        data = json.loads(raw)
        assert data["count"] == 0
        assert data["poses"] == []


# ---------------------------------------------------------------------------
# Tests: get_pose_by_name
# ---------------------------------------------------------------------------

class TestGetPoseByName:

    def test_returns_correct_pose_for_valid_name(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("sala")

        data = json.loads(raw)
        assert data["name"] == "sala"
        assert data["position"]["x"] == pytest.approx(6.94)
        assert data["position"]["y"] == pytest.approx(-9.21)

    def test_returns_error_for_unknown_name(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("nonexistent")

        data = json.loads(raw)
        assert "error" in data
        assert "nonexistent" in data["error"]

    def test_available_poses_listed_on_miss(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("missing")

        data = json.loads(raw)
        assert "available_poses" in data
        assert set(data["available_poses"]) == {"Banco", "sala", "inizio camera"}

    def test_name_lookup_is_case_sensitive(self, saved_poses_mcp):
        """'banco' (lower-case) should NOT match 'Banco'."""
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("banco")

        data = json.loads(raw)
        assert "error" in data

    def test_returns_error_when_no_message(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = None

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("Banco")

        data = json.loads(raw)
        assert "error" in data

    def test_multiword_name_lookup(self, saved_poses_mcp):
        """Pose names with spaces ('inizio camera') should be findable."""
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("inizio camera")

        data = json.loads(raw)
        assert data["name"] == "inizio camera"
        assert data["position"]["x"] == pytest.approx(4.36)

    def test_returned_pose_has_all_fields(self, saved_poses_mcp):
        mock_node = MagicMock()
        mock_node._topic = "/saved_poses"
        mock_node.get_latest.return_value = SAMPLE_PATH

        with patch.object(saved_poses_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = saved_poses_mcp.get_pose_by_name("Banco")

        data = json.loads(raw)
        assert "name" in data
        assert "position" in data
        assert "orientation" in data
        assert set(data["position"]) == {"x", "y", "z"}
        assert set(data["orientation"]) == {"x", "y", "z", "w"}

    def test_reinitializes_node_on_topic_change(self, saved_poses_mcp):
        old_node = MagicMock()
        old_node._topic = "/saved_poses"
        old_node.get_latest.return_value = None

        with patch.object(saved_poses_mcp, "_ros_node", old_node), \
             patch.object(saved_poses_mcp, "_init_ros") as mock_init, \
             patch("time.sleep"):
            saved_poses_mcp.get_pose_by_name("Banco", topic="/other_poses")

        mock_init.assert_called_once_with("/other_poses")


# ---------------------------------------------------------------------------
# Tests: SavedPosesNode callback
# ---------------------------------------------------------------------------

class TestSavedPosesNodeCallback:
    """Tests for the ROS subscriber node's _callback method."""

    def _make_node(self, saved_poses_mcp):
        """Instantiate SavedPosesNode bypassing __init__ ROS calls."""
        node = saved_poses_mcp.SavedPosesNode.__new__(saved_poses_mcp.SavedPosesNode)
        node._lock = threading.Lock()
        node._latest = None
        node._topic = "/saved_poses"
        node.get_logger = MagicMock(return_value=MagicMock())
        return node

    def test_callback_stores_message(self, saved_poses_mcp):
        node = self._make_node(saved_poses_mcp)
        node._callback(SAMPLE_PATH)
        assert node._latest is SAMPLE_PATH

    def test_get_latest_returns_none_before_first_message(self, saved_poses_mcp):
        node = self._make_node(saved_poses_mcp)
        assert node.get_latest() is None

    def test_get_latest_returns_last_received_message(self, saved_poses_mcp):
        node = self._make_node(saved_poses_mcp)
        first_path = _make_path(poses=[_make_pose("A", 1.0, 2.0)])
        second_path = _make_path(poses=[_make_pose("B", 3.0, 4.0)])

        node._callback(first_path)
        node._callback(second_path)

        assert node.get_latest() is second_path

    def test_callback_is_thread_safe(self, saved_poses_mcp):
        """Concurrent writes and reads should never raise."""
        node = self._make_node(saved_poses_mcp)
        errors = []

        def writer():
            for i in range(50):
                node._callback(_make_path(poses=[_make_pose(f"P{i}", float(i), 0.0)]))

        def reader():
            for _ in range(50):
                try:
                    node.get_latest()
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == [], f"Thread safety errors: {errors}"
