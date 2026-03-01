#!/usr/bin/env python3
"""
test_urdf_mcp.py

Unit tests for mcptoollib/urdf_mcp.py MCP server.

All ROS 2 and multiprocessing dependencies are mocked so the tests run
without a live ROS environment.

Run with:
    pytest test/test_urdf_mcp.py -v
"""

import os
import sys
import time
import tempfile
import importlib
import types
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy ROS / MCP dependencies BEFORE importing the module under test
# ---------------------------------------------------------------------------

def _make_ros_stubs():
    """Return a minimal set of stub modules that satisfy urdf_mcp imports."""

    # rclpy
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = MagicMock()
    rclpy_mod.ok = MagicMock(return_value=True)
    rclpy_mod.shutdown = MagicMock()
    rclpy_mod.spin_once = MagicMock()

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object  # base class – subclassed by URDFSubscriberNode

    rclpy_qos = types.ModuleType("rclpy.qos")
    for cls_name in ("QoSProfile", "ReliabilityPolicy", "DurabilityPolicy", "HistoryPolicy"):
        setattr(rclpy_qos, cls_name, MagicMock())

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = MagicMock()

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

        def run(self):
            pass

    mcp_fastmcp.FastMCP = _FastMCP

    return {
        "rclpy": rclpy_mod,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "mcp": mcp_mod,
        "mcp.server": mcp_server,
        "mcp.server.fastmcp": mcp_fastmcp,
    }


@pytest.fixture(autouse=True, scope="module")
def patch_ros_imports():
    """Inject ROS stubs into sys.modules for the duration of the test module."""
    stubs = _make_ros_stubs()
    with mock.patch.dict(sys.modules, stubs):
        yield


# ---------------------------------------------------------------------------
# Import the module under test (after stubs are in place)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def urdf_mcp():
    """Import (or re-import) urdf_mcp with stubs active."""
    stubs = _make_ros_stubs()
    with mock.patch.dict(sys.modules, stubs):
        import mcptoollib.urdf_mcp as mod
        importlib.reload(mod)
        return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
</robot>"""


def _make_cache_with_urdf(urdf_content: str = SAMPLE_URDF):
    """Return a plain dict that mimics the shared-memory cache with URDF data."""
    return {"urdf": urdf_content, "timestamp": time.time()}


def _make_empty_cache():
    return {}


# ---------------------------------------------------------------------------
# Tests: start_urdf_subscriber
# ---------------------------------------------------------------------------

class TestStartUrdfSubscriber:

    def test_starts_new_subscriber(self, urdf_mcp):
        """Should spawn a new process and return a success message."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 1234

        with patch.object(urdf_mcp, "_URDF_PROCESS", None), \
             patch.object(urdf_mcp, "_get_cache", return_value={}), \
             patch.object(urdf_mcp._MP_CONTEXT, "Process", return_value=mock_process) as mock_proc_cls:

            result = urdf_mcp.start_urdf_subscriber("/robot_description")

        mock_process.start.assert_called_once()
        assert "Success" in result
        assert "/robot_description" in result
        assert "1234" in result

    def test_returns_error_if_already_running(self, urdf_mcp):
        """Should refuse to start a second subscriber."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process):
            result = urdf_mcp.start_urdf_subscriber()

        assert "Error" in result
        assert "already running" in result

    def test_custom_topic(self, urdf_mcp):
        """Should pass the custom topic to the worker process."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 42

        with patch.object(urdf_mcp, "_URDF_PROCESS", None), \
             patch.object(urdf_mcp, "_get_cache", return_value={}), \
             patch.object(urdf_mcp._MP_CONTEXT, "Process", return_value=mock_process) as mock_proc_cls:

            result = urdf_mcp.start_urdf_subscriber("/custom_topic")

        _, kwargs = mock_proc_cls.call_args
        assert "/custom_topic" in kwargs.get("args", ())
        assert "/custom_topic" in result


# ---------------------------------------------------------------------------
# Tests: stop_urdf_subscriber
# ---------------------------------------------------------------------------

class TestStopUrdfSubscriber:

    def test_stops_running_subscriber(self, urdf_mcp):
        """Should terminate the process and return a success message."""
        mock_process = MagicMock()
        mock_process.is_alive.side_effect = [True, False]  # alive → stopped

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process):
            result = urdf_mcp.stop_urdf_subscriber()

        mock_process.terminate.assert_called_once()
        mock_process.join.assert_called()
        assert "Success" in result

    def test_returns_info_if_not_running(self, urdf_mcp):
        """Should report that no subscriber is running."""
        with patch.object(urdf_mcp, "_URDF_PROCESS", None):
            result = urdf_mcp.stop_urdf_subscriber()

        assert "Info" in result
        assert "No URDF subscriber" in result

    def test_kills_process_if_terminate_times_out(self, urdf_mcp):
        """Should escalate to kill() if the process does not terminate in time."""
        mock_process = MagicMock()
        # is_alive: first call (running check) → True; after join → still alive
        mock_process.is_alive.side_effect = [True, True, False]

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process):
            result = urdf_mcp.stop_urdf_subscriber()

        mock_process.kill.assert_called_once()
        assert "Success" in result


# ---------------------------------------------------------------------------
# Tests: get_urdf
# ---------------------------------------------------------------------------

class TestGetUrdf:

    def test_returns_urdf_when_cached(self, urdf_mcp):
        cache = _make_cache_with_urdf()
        with patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.get_urdf()

        assert result == SAMPLE_URDF

    def test_returns_error_when_cache_empty(self, urdf_mcp):
        with patch.object(urdf_mcp, "_get_cache", return_value=_make_empty_cache()):
            result = urdf_mcp.get_urdf()

        assert "Error" in result
        assert "No URDF available" in result


# ---------------------------------------------------------------------------
# Tests: save_urdf_to_file
# ---------------------------------------------------------------------------

class TestSaveUrdfToFile:

    def test_saves_urdf_to_specified_path(self, urdf_mcp, tmp_path):
        filepath = str(tmp_path / "robot.urdf")
        cache = _make_cache_with_urdf()

        with patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.save_urdf_to_file(filepath)

        assert "Success" in result
        assert filepath in result
        with open(filepath) as f:
            assert f.read() == SAMPLE_URDF

    def test_returns_error_when_cache_empty(self, urdf_mcp, tmp_path):
        filepath = str(tmp_path / "robot.urdf")
        with patch.object(urdf_mcp, "_get_cache", return_value=_make_empty_cache()):
            result = urdf_mcp.save_urdf_to_file(filepath)

        assert "Error" in result
        assert not os.path.exists(filepath)

    def test_returns_error_on_write_failure(self, urdf_mcp):
        cache = _make_cache_with_urdf()
        bad_path = "/nonexistent_root/deep/path/robot.urdf"

        with patch.object(urdf_mcp, "_get_cache", return_value=cache), \
             patch("builtins.open", side_effect=PermissionError("denied")):
            result = urdf_mcp.save_urdf_to_file(bad_path)

        assert "Error" in result

    def test_creates_intermediate_directories(self, urdf_mcp, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c" / "robot.urdf")
        cache = _make_cache_with_urdf()

        with patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.save_urdf_to_file(nested)

        assert "Success" in result
        assert os.path.exists(nested)


# ---------------------------------------------------------------------------
# Tests: get_temp_urdf_path
# ---------------------------------------------------------------------------

class TestGetTempUrdfPath:

    def test_returns_path_of_temp_file(self, urdf_mcp):
        cache = _make_cache_with_urdf()
        with patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.get_temp_urdf_path()

        assert result.startswith("Success:")
        tmp_path = result.split("Success: ", 1)[1].strip()
        assert os.path.isfile(tmp_path)
        assert tmp_path.endswith(".urdf")

        with open(tmp_path) as f:
            assert f.read() == SAMPLE_URDF

        os.unlink(tmp_path)  # cleanup

    def test_returns_error_when_cache_empty(self, urdf_mcp):
        with patch.object(urdf_mcp, "_get_cache", return_value=_make_empty_cache()):
            result = urdf_mcp.get_temp_urdf_path()

        assert "Error" in result
        assert "No URDF available" in result

    def test_returns_error_on_tempfile_failure(self, urdf_mcp):
        cache = _make_cache_with_urdf()
        with patch.object(urdf_mcp, "_get_cache", return_value=cache), \
             patch("tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
            result = urdf_mcp.get_temp_urdf_path()

        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests: urdf_status
# ---------------------------------------------------------------------------

class TestUrdfStatus:

    def test_status_not_started(self, urdf_mcp):
        with patch.object(urdf_mcp, "_URDF_PROCESS", None), \
             patch.object(urdf_mcp, "_get_cache", return_value=_make_empty_cache()):
            result = urdf_mcp.urdf_status()

        assert "Not started" in result
        assert "Empty" in result

    def test_status_running_with_urdf(self, urdf_mcp):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 999
        cache = _make_cache_with_urdf()

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process), \
             patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.urdf_status()

        assert "Running" in result
        assert "999" in result
        assert "Available" in result

    def test_status_stopped_with_urdf(self, urdf_mcp):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        cache = _make_cache_with_urdf()

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process), \
             patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.urdf_status()

        assert "Stopped" in result
        assert "Available" in result

    def test_status_running_no_urdf(self, urdf_mcp):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 7

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process), \
             patch.object(urdf_mcp, "_get_cache", return_value=_make_empty_cache()):
            result = urdf_mcp.urdf_status()

        assert "Running" in result
        assert "Empty" in result

    def test_status_includes_timestamp(self, urdf_mcp):
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 5
        cache = _make_cache_with_urdf()

        with patch.object(urdf_mcp, "_URDF_PROCESS", mock_process), \
             patch.object(urdf_mcp, "_get_cache", return_value=cache):
            result = urdf_mcp.urdf_status()

        assert "Last Updated" in result


# ---------------------------------------------------------------------------
# Tests: URDFSubscriberNode callback
# ---------------------------------------------------------------------------

class TestURDFSubscriberNodeCallback:
    """Tests for the ROS subscriber node's urdf_callback method."""

    def _make_node(self, urdf_mcp):
        """Instantiate URDFSubscriberNode with mocked ROS internals."""
        cache = {}

        node = urdf_mcp.URDFSubscriberNode.__new__(urdf_mcp.URDFSubscriberNode)
        node.cache = cache
        node.get_logger = MagicMock(return_value=MagicMock())
        return node, cache

    def test_valid_urdf_stored_in_cache(self, urdf_mcp):
        node, cache = self._make_node(urdf_mcp)
        msg = MagicMock()
        msg.data = SAMPLE_URDF

        node.urdf_callback(msg)

        assert cache["urdf"] == SAMPLE_URDF
        assert "timestamp" in cache

    def test_invalid_urdf_not_stored(self, urdf_mcp):
        node, cache = self._make_node(urdf_mcp)
        msg = MagicMock()
        msg.data = "not a valid urdf"

        node.urdf_callback(msg)

        assert "urdf" not in cache

    def test_empty_message_not_stored(self, urdf_mcp):
        node, cache = self._make_node(urdf_mcp)
        msg = MagicMock()
        msg.data = ""

        node.urdf_callback(msg)

        assert "urdf" not in cache

    def test_timestamp_is_recent(self, urdf_mcp):
        node, cache = self._make_node(urdf_mcp)
        msg = MagicMock()
        msg.data = SAMPLE_URDF
        before = time.time()

        node.urdf_callback(msg)

        after = time.time()
        assert before <= cache["timestamp"] <= after
