"""Saved frames carry their depth (perception_node._save_frame), and each message
the time its frame took in the detector and the pose extractor."""
import queue
import time

import cv2
import numpy as np

from src.perception.camera import MockCamera
from src.perception.perception_node import PerceptionNode


class _NoDetector:
    def detect(self, rgb):
        return []


def test_the_depth_is_saved_beside_the_rgb_and_reads_back_exactly(tmp_path):
    rng = np.random.default_rng(0)
    depth = (rng.integers(300, 1500, (720, 1280)) / 1000.0).astype(np.float32)   # z16 * 0.001 m
    depth[:10] = 0.0                                                              # holes stay holes
    rgb = np.zeros((720, 1280, 3), np.uint8)
    node = PerceptionNode(MockCamera([rgb], [depth]), _NoDetector(), queue.Queue(),
                          save_frames_dir=str(tmp_path))
    msg = node.process_once()
    p = msg["frame_path"]
    back = cv2.imread(p.replace(".png", "_depth.png"), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16
    assert np.array_equal(back.astype(np.float32) / 1000.0, depth)


class _SlowDetector:
    def detect(self, rgb):
        time.sleep(0.05)
        return []


def test_the_message_times_the_detector_apart_from_the_extractor():
    rgb = np.zeros((720, 1280, 3), np.uint8)
    depth = np.ones((720, 1280), np.float32)
    msg = PerceptionNode(MockCamera([rgb], [depth]), _SlowDetector(),
                         queue.Queue()).process_once()
    assert msg["inference_ms"] >= 50.0
    assert 0.0 <= msg["localize_ms"] < msg["inference_ms"]
