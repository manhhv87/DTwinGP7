"""The end of a job as _wait_idle times it (MotomanHSEBackend.last_job_timing).

The controller is a scripted one behind the socket, and the backend module's
`time` a clock that advances only with the exchanges and the sleeps, so the instant
Running clears is known exactly. Each reply carries the controller's state at the
middle of its exchange and arrives one round trip (10.2 ms, as measured on the
cell) after the request.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from src.orchestrator.backends import motoman_hse as mh
from src.orchestrator.backends.hse_protocol import (
    ACK_RESPONSE, HSE_HEADER_SIZE, HSE_IDENTIFIER, HSE_RESERVE1, HSE_RESERVE2,
    Command,
)
from src.orchestrator.backends.motoman_hse import MotionHaltedError, MotomanHSEBackend

RTT = 0.0102
T0 = 1000.0


class Clock:
    """Stands in for the time module of motoman_hse."""

    def __init__(self):
        self.t = T0

    def perf_counter(self):
        return self.t

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def __getattr__(self, name):          # strftime and the rest: the real ones
        return getattr(time, name)


def _reply(payload: bytes, request_id: int, status: int = 0) -> bytes:
    header = struct.pack("<4s H H B B B B I 8s B B B B H H", HSE_IDENTIFIER,
                         HSE_HEADER_SIZE, len(payload), HSE_RESERVE1, 0x01,
                         ACK_RESPONSE, request_id, 0, HSE_RESERVE2,
                         0x01, status, 0, 0, 0, 0)
    return header + payload


class FakeYRC:
    """A controller running its job from `rise` to `fall`, with Running cleared
    and a hold bit set inside `hold`; no alarm. timeout_at drops the reply of
    that exchange (0-based); wrong_id_at echoes another request_id on that one;
    error_at answers that one with an error status;
    hold_for_one_read releases the hold right after the first read that saw it,
    before its follow-up, as an operator releasing HOLD at that instant would."""

    def __init__(self, clock, rise, fall, hold=None, timeout_at=None, wrong_id_at=None,
                 hold_for_one_read=False, error_at=None):
        self.clock, self.rise, self.fall, self.hold = clock, rise, fall, hold
        self.timeout_at, self.wrong_id_at, self.error_at = timeout_at, wrong_id_at, error_at
        self.hold_for_one_read = hold_for_one_read
        self.sent, self.blocking = [], True

    def held(self, t):
        return self.hold is not None and self.hold[0] <= t < self.hold[1]

    def sendto(self, packet, addr):
        self.sent.append((packet, self.clock.t))

    def setblocking(self, flag):
        self.blocking = flag

    def settimeout(self, s):
        pass

    def recvfrom(self, n):
        if not self.blocking:
            raise BlockingIOError
        packet, t_send = self.sent[-1]
        k = len(self.sent) - 1
        if k == self.timeout_at:
            self.clock.t = t_send + 0.5
            raise socket.timeout
        command = struct.unpack_from("<H", packet, 24)[0]
        request_id = packet[11]
        t = t_send + RTT / 2
        self.clock.t = t_send + RTT
        payload = b"\x00" * 8
        if command == Command.READ_STATUS:
            held = self.held(t)
            if held and self.hold_for_one_read:
                self.hold = (self.hold[0], t + 0.75 * RTT)
                self.hold_for_one_read = False
            running = self.rise <= t < self.fall and not held
            payload = bytes([0x08 if running else 0x00, 0, 0, 0,
                             0x02 if held else 0x00, 0, 0, 0])
        if k == self.wrong_id_at:
            request_id = (request_id + 7) % 256
        status = 0x1F if k == self.error_at else 0
        return _reply(payload, request_id, status), ("192.168.1.100", 10040)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(mh, "time", c)
    return c


def _backend(yrc):
    b = MotomanHSEBackend(ip="192.168.1.100", timeout_s=0.5)
    b._sock = yrc
    b.start_confirm_timeout_s = 2.0
    b.wait_completion_timeout_s = 60.0
    return b


def test_the_end_lies_in_its_bracket_and_the_midpoint_is_unbiased(clock):
    errors, late = [], []
    period, n = 0.1 + RTT, 110            # phase 2: a 0.1 s sleep and one read
    for k in range(n):
        clock.t = T0
        fall = T0 + 9.0 + period * k / n                        # one poll period, evenly
        b = _backend(FakeYRC(clock, rise=T0 + 0.08, fall=fall))
        b._wait_idle()
        timing = b.last_job_timing()
        assert timing["flags"] == []
        assert timing["lo"] < fall <= timing["hi"]
        half = 0.5 * (timing["hi"] - timing["lo"])
        assert abs(timing["end"] - fall) <= half
        errors.append(timing["end"] - fall)
        late.append(clock.t - fall)                             # where the wait returned
    assert abs(sum(errors) / len(errors)) < 0.002
    assert max(abs(e) for e in errors) < 0.070
    assert min(late) > 2 * RTT                                  # the old stop, always late
    assert sum(late) / len(late) > 0.07                         # and late by ~0.08 s on average


def test_a_hold_moves_the_bracket_after_it_and_is_flagged(clock):
    fall = T0 + 12.0
    b = _backend(FakeYRC(clock, rise=T0 + 0.05, fall=fall, hold=(T0 + 4.0, T0 + 6.5)))
    b._wait_idle()
    timing = b.last_job_timing()
    assert timing["flags"] == ["held"]
    assert T0 + 6.5 < timing["lo"] < fall <= timing["hi"]


def test_a_hold_released_between_the_two_reads_leaves_no_end(clock):
    """The wait ends on a follow-up that saw no hold while the first read saw
    one; the job may have resumed, so it is flagged and not timed."""
    b = _backend(FakeYRC(clock, rise=T0 + 0.05, fall=T0 + 12.0, hold=(T0 + 4.0, T0 + 99.0),
                         hold_for_one_read=True))
    b._wait_idle()
    timing = b.last_job_timing()
    assert timing["end"] is None and timing["flags"] == ["held", "race"]


def test_a_hold_inside_the_start_latency_is_flagged(clock):
    b = _backend(FakeYRC(clock, rise=T0 + 1.0, fall=T0 + 9.0, hold=(T0 + 0.2, T0 + 0.6)))
    b._wait_idle()
    assert b.last_job_timing()["flags"] == ["held"]


def test_a_failed_read_during_the_job_is_flagged(clock):
    b = _backend(FakeYRC(clock, rise=T0 + 0.05, fall=T0 + 9.0, error_at=25))
    b._wait_idle()
    timing = b.last_job_timing()
    assert timing["flags"] == ["error"] and timing["end"] is not None


def test_a_timed_out_read_during_the_job_is_flagged(clock):
    yrc = FakeYRC(clock, rise=T0 + 0.05, fall=T0 + 9.0, timeout_at=20)
    b = _backend(yrc)
    b._wait_idle()
    timing = b.last_job_timing()
    assert "timeout" in timing["flags"] and timing["end"] is not None
    assert timing["lo"] < yrc.fall <= timing["hi"]


def test_a_mismatched_reply_during_the_job_is_flagged(clock):
    b = _backend(FakeYRC(clock, rise=T0 + 0.05, fall=T0 + 9.0, wrong_id_at=15))
    b._wait_idle()
    assert b.last_job_timing()["flags"] == ["mismatch"]


def test_another_threads_exchanges_leave_this_threads_record(clock):
    b = _backend(FakeYRC(clock, rise=T0 + 0.05, fall=T0 + 9.0))
    b._wait_idle()
    timing, exchange = b.last_job_timing(), b.last_exchange()
    worker = threading.Thread(target=b.read_status_running)    # e.g. the mirror
    worker.start()
    worker.join()
    assert b.last_job_timing() is timing and b.last_exchange() == exchange


def test_a_dispatch_clears_the_timing_of_the_job_before(clock):
    b = _backend(FakeYRC(clock, rise=T0 + 0.05, fall=T0 + 9.0))
    b._wait_idle()
    assert b.last_job_timing()["end"] is not None
    b._halted = True                                           # refused before anything is sent
    with pytest.raises(MotionHaltedError):
        b.run_dtwinrun_motion(None, None, None)
    assert b.last_job_timing() is None
