"""The capture instant of a camera frame on the perf_counter axis (camera.frame_clock).

Frames are stand-ins with the pyrealsense2 calls frame_clock makes; the system
clock reads S_MS; of the three perf_counter pairs read around it the narrowest is
100.000000 and 100.000002, so a system time x ms before S_MS lands x ms before
100.000001 on the perf_counter axis (the wider pairs, widened by another thread,
would put it elsewhere).
"""
import types

import pytest

from src.perception.camera import frame_clock

S_MS = 1.8e12
MID = 100.000001
MD = types.SimpleNamespace(time_of_arrival="toa", frame_timestamp="fts", sensor_timestamp="sts")
RS = types.SimpleNamespace(frame_metadata_value=MD)


class _Domain:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"timestamp_domain.{self.name}"


class _Frame:
    def __init__(self, domain, ts_ms, **md):
        self._domain, self._ts, self._md = _Domain(domain), ts_ms, md

    def get_frame_timestamp_domain(self):
        return self._domain

    def get_timestamp(self):
        return self._ts

    def supports_frame_metadata(self, key):
        return key in self._md

    def get_frame_metadata(self, key):
        return self._md[key]


class _Frames:
    def __init__(self, color):
        self._c = color

    def get_color_frame(self):
        return self._c


def _stamp(color):
    ticks = iter((99.9, 100.1, 100.0, 100.000002, 100.3, 100.31))
    return frame_clock(_Frames(color), t_got=100.0, rs=RS, now_ms=lambda: S_MS,
                       perf=lambda: next(ticks))


def test_global_time_is_moved_back_to_mid_exposure():
    # readout began 80 ms before the clock pair; mid-exposure 12 ms before readout
    out = _stamp(_Frame("global_time", S_MS - 80.0, fts=1_000_000, sts=988_000,
                        toa=S_MS - 60.0))
    assert out["src"] == "global_mid"
    assert out["capture"] == pytest.approx(MID - 0.092, abs=1e-9)
    assert out["arrival"] == pytest.approx(MID - 0.060, abs=1e-9)


def test_global_time_without_the_sensor_timestamp_keeps_the_readout():
    out = _stamp(_Frame("global_time", S_MS - 80.0, toa=S_MS - 60.0))
    assert out["src"] == "global_readout"
    assert out["capture"] == pytest.approx(MID - 0.080, abs=1e-9)


def test_without_metadata_the_host_arrival_stands_for_the_capture():
    out = _stamp(_Frame("system_time", S_MS - 60.0, toa=S_MS - 60.0))
    assert out["src"] == "arrival" and out["domain"] == "system_time"
    assert out["capture"] == pytest.approx(MID - 0.060, abs=1e-9)


def test_without_an_arrival_the_receipt_stands_for_it():
    out = _stamp(_Frame("hardware_clock", 123456.0))
    assert out["src"] == "got" and out["capture"] == 100.0 and out["arrival"] is None


def test_the_names_exist_in_pyrealsense2():
    rs = pytest.importorskip("pyrealsense2")
    for key in ("time_of_arrival", "frame_timestamp", "sensor_timestamp"):
        assert hasattr(rs.frame_metadata_value, key)
    assert str(rs.timestamp_domain.global_time).rsplit(".", 1)[-1] == "global_time"
