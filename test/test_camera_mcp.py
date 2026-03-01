#!/usr/bin/env python3
"""
test_camera_mcp.py

Unit tests for mcptoollib/camera_mcp.py MCP server.

All ROS 2 dependencies are mocked so the tests run without a live ROS
environment.  The module-level _init_ros() / spin thread is also bypassed
so the import itself is side-effect-free.

Run with:
    pytest test/test_camera_mcp.py -v
"""

import base64
import importlib
import json
import sys
import threading
import time
import types
from unittest import mock
import io
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ROS / MCP stubs — injected before the module under test is imported
# ---------------------------------------------------------------------------

def _make_stubs():
    """Return a minimal set of stub modules that satisfy camera_mcp imports."""

    # rclpy
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = MagicMock()
    rclpy_mod.ok = MagicMock(return_value=True)
    rclpy_mod.shutdown = MagicMock()
    rclpy_mod.spin = MagicMock()

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

    # sensor_msgs
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class _CompressedImage:
        """Minimal sensor_msgs/CompressedImage stub."""
        def __init__(self):
            self.header = _Header()
            self.format = "jpeg"
            self.data = b""

    class _Header:
        def __init__(self):
            self.stamp = _Stamp()
            self.frame_id = ""

    class _Stamp:
        def __init__(self):
            self.sec = 0
            self.nanosec = 0

    sensor_msgs_msg.CompressedImage = _CompressedImage

    # mcp
    mcp_mod = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    mcp_fastmcp = types.ModuleType("mcp.server.fastmcp")

    class _FastMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            """Passthrough decorator — does not wrap the function."""
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
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "mcp": mcp_mod,
        "mcp.server": mcp_server,
        "mcp.server.fastmcp": mcp_fastmcp,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def patch_ros_imports():
    """Keep ROS stubs injected for the entire test module."""
    stubs = _make_stubs()
    with mock.patch.dict(sys.modules, stubs):
        yield


@pytest.fixture(scope="module")
def camera_mcp():
    """
    Import (or reload) camera_mcp with all ROS stubs active and the
    module-level _init_ros() call patched to a no-op.
    """
    stubs = _make_stubs()
    dummy_thread = MagicMock(spec=threading.Thread)

    with mock.patch.dict(sys.modules, stubs), \
         mock.patch("threading.Thread", return_value=dummy_thread):
        import mcptoollib.camera_mcp as mod
        importlib.reload(mod)
        return mod


# ---------------------------------------------------------------------------
# Helpers: build fake CompressedImage messages
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers: generate minimal real JPEG/PNG bytes using Pillow
# (real Pillow is installed; no need to stub it)
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage
    def _make_jpeg_bytes(width: int = 800, height: int = 600) -> bytes:
        buf = io.BytesIO()
        _PILImage.new("RGB", (width, height), color=(100, 150, 200)).save(buf, format="JPEG")
        return buf.getvalue()
    def _make_png_bytes(width: int = 800, height: int = 600) -> bytes:
        buf = io.BytesIO()
        _PILImage.new("RGB", (width, height), color=(200, 100, 50)).save(buf, format="PNG")
        return buf.getvalue()
    SAMPLE_JPEG_DATA = _make_jpeg_bytes(800, 600)   # larger than 640x480 -> will be resized
    SAMPLE_JPEG_DATA_SMALL = _make_jpeg_bytes(320, 240)  # already fits -> no resize
    SAMPLE_PNG_DATA = _make_png_bytes(800, 600)
except ImportError:
    # Fallback if Pillow is somehow unavailable (keeps stubs working)
    SAMPLE_JPEG_DATA = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    SAMPLE_JPEG_DATA_SMALL = SAMPLE_JPEG_DATA
    SAMPLE_PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"\x00" * 80


def _make_compressed_image(
    fmt: str = "jpeg",
    data: bytes = SAMPLE_JPEG_DATA,
    frame_id: str = "camera_color_optical_frame",
    sec: int = 1772368852,
):
    """Return a minimal CompressedImage-like object."""
    class _Stamp:
        pass
    class _Header:
        pass
    class _Msg:
        pass

    stamp = _Stamp()
    stamp.sec = sec
    stamp.nanosec = 0

    header = _Header()
    header.stamp = stamp
    header.frame_id = frame_id

    msg = _Msg()
    msg.header = header
    msg.format = fmt
    msg.data = data
    return msg


SAMPLE_IMAGE = _make_compressed_image()


# ---------------------------------------------------------------------------
# Tests: _init_ros
# ---------------------------------------------------------------------------

class TestInitRos:

    def test_initializes_node_and_thread(self, camera_mcp):
        """_init_ros() should create a CameraNode and start a daemon thread."""
        mock_node = MagicMock()
        mock_thread = MagicMock()
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True

        with patch.object(camera_mcp, "_ros_node", None), \
             patch.object(camera_mcp, "_spin_thread", None), \
             patch.object(camera_mcp, "rclpy", mock_rclpy), \
             patch.object(camera_mcp, "CameraNode", return_value=mock_node) as node_cls, \
             patch("threading.Thread", return_value=mock_thread):

            camera_mcp._init_ros("/test_topic")

        node_cls.assert_called_once_with("/test_topic")
        mock_thread.start.assert_called_once()

    def test_skips_reinit_if_node_already_set(self, camera_mcp):
        """_init_ros() is a no-op when a node is already initialized."""
        existing_node = MagicMock()

        with patch.object(camera_mcp, "_ros_node", existing_node), \
             patch.object(camera_mcp, "CameraNode") as node_cls:

            camera_mcp._init_ros("/test_topic")

        node_cls.assert_not_called()

    def test_calls_rclpy_init_when_not_ok(self, camera_mcp):
        """Should call rclpy.init() when rclpy is not yet initialized."""
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = False
        mock_node = MagicMock()

        with patch.object(camera_mcp, "_ros_node", None), \
             patch.object(camera_mcp, "_spin_thread", None), \
             patch.object(camera_mcp, "rclpy", mock_rclpy), \
             patch.object(camera_mcp, "CameraNode", return_value=mock_node), \
             patch("threading.Thread", return_value=MagicMock()):

            camera_mcp._init_ros()

        mock_rclpy.init.assert_called()


# ---------------------------------------------------------------------------
# Tests: get_camera_image
# ---------------------------------------------------------------------------

class TestGetCameraImage:

    def test_returns_base64_encoded_image(self, camera_mcp):
        """Output base64 should be whatever _resize_and_encode returns."""
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"fake-resized-jpeg").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))) as mock_resize, \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        mock_resize.assert_called_once_with(
            SAMPLE_IMAGE.data,
            camera_mcp._DEFAULT_MAX_WIDTH,
            camera_mcp._DEFAULT_MAX_HEIGHT,
        )
        data = json.loads(raw)
        assert data["base64"] == fake_b64

    def test_returns_correct_format(self, camera_mcp):
        """format field must always be 'jpeg' after resize."""
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"x").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        assert data["format"] == "jpeg"

    def test_returns_size_bytes(self, camera_mcp):
        """size_bytes must reflect the resized JPEG byte count, not the raw input."""
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        resized_payload = b"resized-jpeg-bytes"
        fake_b64 = base64.b64encode(resized_payload).decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        assert data["size_bytes"] == len(resized_payload)

    def test_returns_topic_in_output(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"x").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        assert data["topic"] == camera_mcp._DEFAULT_TOPIC

    def test_returns_error_json_when_no_image(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = None

        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        assert "error" in data

    def test_error_message_mentions_topic(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = None

        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        assert camera_mcp._DEFAULT_TOPIC in data["error"]

    def test_handles_png_format(self, camera_mcp):
        """PNG input is always re-encoded to JPEG by _resize_and_encode."""
        png_image = _make_compressed_image(fmt="png", data=SAMPLE_PNG_DATA)
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = png_image

        fake_b64 = base64.b64encode(b"jpeg-from-png").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        # format is always "jpeg" after resize/re-encode
        assert data["format"] == "jpeg"
        assert data["base64"] == fake_b64

    def test_reinitializes_node_on_topic_change(self, camera_mcp):
        """When a different topic is requested the node should be re-initialized."""
        old_node = MagicMock()
        old_node._topic = camera_mcp._DEFAULT_TOPIC
        old_node.get_latest.return_value = None

        with patch.object(camera_mcp, "_ros_node", old_node), \
             patch.object(camera_mcp, "_init_ros") as mock_init, \
             patch("time.sleep"):
            camera_mcp.get_camera_image(topic="/other_camera")

        mock_init.assert_called_once_with("/other_camera")

    def test_output_is_valid_json(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"x").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        # Must not raise
        json.loads(raw)

    def test_base64_is_ascii_safe(self, camera_mcp):
        """Ensure base64 output contains only ASCII characters."""
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"x").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        data["base64"].encode("ascii")  # Must not raise

    def test_returns_original_and_resized_size(self, camera_mcp):
        """Response must include original_size and resized_size dicts."""
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"x").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (640, 450))), \
             patch("time.sleep"):
            raw = camera_mcp.get_camera_image()

        data = json.loads(raw)
        assert data["original_size"] == {"width": 800, "height": 600}
        assert data["resized_size"] == {"width": 640, "height": 450}

    def test_passes_max_width_height_to_resize(self, camera_mcp):
        """max_width and max_height params must be forwarded to _resize_and_encode."""
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE

        fake_b64 = base64.b64encode(b"x").decode("ascii")
        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch.object(camera_mcp, "_resize_and_encode",
                          return_value=(fake_b64, (800, 600), (320, 240))) as mock_resize, \
             patch("time.sleep"):
            camera_mcp.get_camera_image(max_width=320, max_height=240)

        mock_resize.assert_called_once_with(SAMPLE_IMAGE.data, 320, 240)


# ---------------------------------------------------------------------------
# Tests: _resize_and_encode (pure function, no ROS needed)
# ---------------------------------------------------------------------------

class TestResizeAndEncode:
    """Tests for the _resize_and_encode helper.  Real Pillow is used."""

    def test_returns_tuple_of_three(self, camera_mcp):
        result = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA)
        assert isinstance(result, tuple) and len(result) == 3

    def test_base64_is_valid(self, camera_mcp):
        b64_str, _, _ = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA)
        # Should not raise
        decoded = base64.b64decode(b64_str)
        assert len(decoded) > 0

    def test_original_size_matches_input(self, camera_mcp):
        _, original, _ = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA)
        # SAMPLE_JPEG_DATA is 800x600
        assert original == (800, 600)

    def test_large_image_is_downscaled(self, camera_mcp):
        _, _, resized = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA, 640, 480)
        assert resized[0] <= 640
        assert resized[1] <= 480

    def test_small_image_not_upscaled(self, camera_mcp):
        """thumbnail() never upscales; 320x240 stays 320x240 with 640x480 budget."""
        _, original, resized = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA_SMALL, 640, 480)
        assert original == (320, 240)
        assert resized == (320, 240)

    def test_aspect_ratio_preserved(self, camera_mcp):
        """800x600 resized to 640x480 should be 640x480 (exact fit)."""
        _, original, resized = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA, 640, 480)
        orig_ratio = original[0] / original[1]
        res_ratio = resized[0] / resized[1]
        assert abs(orig_ratio - res_ratio) < 0.02  # within 2% tolerance

    def test_output_is_jpeg(self, camera_mcp):
        """Re-encoded output must be a valid JPEG (starts with FF D8)."""
        b64_str, _, _ = camera_mcp._resize_and_encode(SAMPLE_JPEG_DATA)
        decoded = base64.b64decode(b64_str)
        assert decoded[:2] == b"\xff\xd8", "Expected JPEG magic bytes"

    def test_png_input_becomes_jpeg(self, camera_mcp):
        """PNG input should be converted to JPEG on output."""
        b64_str, _, _ = camera_mcp._resize_and_encode(SAMPLE_PNG_DATA)
        decoded = base64.b64decode(b64_str)
        assert decoded[:2] == b"\xff\xd8", "PNG input should produce JPEG output"

# ---------------------------------------------------------------------------
# Tests: camera_status
# ---------------------------------------------------------------------------

class TestCameraStatus:

    def test_returns_status_with_image(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = SAMPLE_IMAGE
        mock_node.get_stats.return_value = {
            "topic": camera_mcp._DEFAULT_TOPIC,
            "messages_received": 42,
            "has_image": True,
            "last_received": 1772368852.0,
        }

        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = camera_mcp.camera_status()

        data = json.loads(raw)
        assert data["topic"] == camera_mcp._DEFAULT_TOPIC
        assert data["messages_received"] == 42
        assert data["has_image"] is True
        assert data["image_format"] == "jpeg"
        assert data["image_size_bytes"] == len(SAMPLE_IMAGE.data)  # raw bytes from msg, not resized

    def test_returns_status_without_image(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = None
        mock_node.get_stats.return_value = {
            "topic": camera_mcp._DEFAULT_TOPIC,
            "messages_received": 0,
            "has_image": False,
            "last_received": None,
        }

        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = camera_mcp.camera_status()

        data = json.loads(raw)
        assert data["has_image"] is False
        assert data["messages_received"] == 0
        assert "image_format" not in data
        assert "image_size_bytes" not in data

    def test_reinitializes_node_on_topic_change(self, camera_mcp):
        old_node = MagicMock()
        old_node._topic = camera_mcp._DEFAULT_TOPIC
        old_node.get_latest.return_value = None
        old_node.get_stats.return_value = {
            "topic": camera_mcp._DEFAULT_TOPIC,
            "messages_received": 0,
            "has_image": False,
            "last_received": None,
        }

        with patch.object(camera_mcp, "_ros_node", old_node), \
             patch.object(camera_mcp, "_init_ros") as mock_init, \
             patch("time.sleep"):
            camera_mcp.camera_status(topic="/other_camera")

        mock_init.assert_called_once_with("/other_camera")

    def test_output_is_valid_json(self, camera_mcp):
        mock_node = MagicMock()
        mock_node._topic = camera_mcp._DEFAULT_TOPIC
        mock_node.get_latest.return_value = None
        mock_node.get_stats.return_value = {
            "topic": camera_mcp._DEFAULT_TOPIC,
            "messages_received": 0,
            "has_image": False,
            "last_received": None,
        }

        with patch.object(camera_mcp, "_ros_node", mock_node), \
             patch("time.sleep"):
            raw = camera_mcp.camera_status()

        json.loads(raw)


# ---------------------------------------------------------------------------
# Tests: CameraNode callback
# ---------------------------------------------------------------------------

class TestCameraNodeCallback:
    """Tests for the ROS subscriber node's _callback method."""

    def _make_node(self, camera_mcp):
        """Instantiate CameraNode bypassing __init__ ROS calls."""
        node = camera_mcp.CameraNode.__new__(camera_mcp.CameraNode)
        node._lock = threading.Lock()
        node._latest = None
        node._topic = camera_mcp._DEFAULT_TOPIC
        node._msg_count = 0
        node._last_recv_time = None
        node.get_logger = MagicMock(return_value=MagicMock())
        return node

    def test_callback_stores_message(self, camera_mcp):
        node = self._make_node(camera_mcp)
        node._callback(SAMPLE_IMAGE)
        assert node._latest is SAMPLE_IMAGE

    def test_get_latest_returns_none_before_first_message(self, camera_mcp):
        node = self._make_node(camera_mcp)
        assert node.get_latest() is None

    def test_get_latest_returns_last_received_message(self, camera_mcp):
        node = self._make_node(camera_mcp)
        first = _make_compressed_image(data=b"\x01\x02")
        second = _make_compressed_image(data=b"\x03\x04")

        node._callback(first)
        node._callback(second)

        assert node.get_latest() is second

    def test_callback_increments_msg_count(self, camera_mcp):
        node = self._make_node(camera_mcp)
        node._callback(SAMPLE_IMAGE)
        node._callback(SAMPLE_IMAGE)
        assert node._msg_count == 2

    def test_callback_updates_recv_time(self, camera_mcp):
        node = self._make_node(camera_mcp)
        assert node._last_recv_time is None
        node._callback(SAMPLE_IMAGE)
        assert node._last_recv_time is not None
        assert isinstance(node._last_recv_time, float)

    def test_get_stats_returns_correct_dict(self, camera_mcp):
        node = self._make_node(camera_mcp)
        node._callback(SAMPLE_IMAGE)

        stats = node.get_stats()
        assert stats["topic"] == camera_mcp._DEFAULT_TOPIC
        assert stats["messages_received"] == 1
        assert stats["has_image"] is True
        assert stats["last_received"] is not None

    def test_get_stats_before_any_message(self, camera_mcp):
        node = self._make_node(camera_mcp)
        stats = node.get_stats()
        assert stats["messages_received"] == 0
        assert stats["has_image"] is False
        assert stats["last_received"] is None

    def test_callback_is_thread_safe(self, camera_mcp):
        """Concurrent writes and reads should never raise."""
        node = self._make_node(camera_mcp)
        errors = []

        def writer():
            for i in range(50):
                node._callback(_make_compressed_image(data=bytes([i % 256])))

        def reader():
            for _ in range(50):
                try:
                    node.get_latest()
                    node.get_stats()
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == [], f"Thread safety errors: {errors}"
