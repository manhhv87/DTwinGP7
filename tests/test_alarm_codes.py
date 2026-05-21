"""
test_alarm_codes.py
───────────────────
Verify alarm code decoder + severity classification.
"""
from __future__ import annotations

import pytest

from src.orchestrator.backends.alarm_codes import (
    AlarmSeverity,
    decode_alarm,
    is_recoverable,
)


class TestDecodeKnownCodes:
    def test_zero_is_no_alarm(self):
        info = decode_alarm(0)
        assert info.severity == AlarmSeverity.NONE
        assert info.name == "NONE"

    def test_2010_is_emergency_stop_major(self):
        info = decode_alarm(2010)
        assert info.severity == AlarmSeverity.MAJOR
        assert "EMERGENCY" in info.name
        assert "E-stop" in info.recovery_hint

    def test_4010_job_not_found_user(self):
        info = decode_alarm(4010)
        assert info.severity == AlarmSeverity.USER
        assert info.code == 4010

    def test_1010_remote_mode_minor(self):
        info = decode_alarm(1010)
        assert info.severity == AlarmSeverity.MINOR


class TestUnknownCodeFallback:
    def test_unknown_in_minor_range(self):
        info = decode_alarm(1500)
        assert info.severity == AlarmSeverity.MINOR
        assert info.name.startswith("UNKNOWN_")
        assert info.code == 1500

    def test_unknown_in_major_range(self):
        info = decode_alarm(2999)
        assert info.severity == AlarmSeverity.MAJOR

    def test_unknown_in_system_range(self):
        info = decode_alarm(3500)
        assert info.severity == AlarmSeverity.SYSTEM

    def test_unknown_in_user_range(self):
        info = decode_alarm(4500)
        assert info.severity == AlarmSeverity.USER


class TestRecoverable:
    def test_none_is_recoverable(self):
        assert is_recoverable(decode_alarm(0))

    def test_minor_is_recoverable(self):
        assert is_recoverable(decode_alarm(1010))

    def test_user_is_recoverable(self):
        assert is_recoverable(decode_alarm(4010))

    def test_major_is_not_recoverable(self):
        assert not is_recoverable(decode_alarm(2010))

    def test_system_is_not_recoverable(self):
        assert not is_recoverable(decode_alarm(3020))
