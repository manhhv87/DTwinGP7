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
    """Unknown codes default FAIL-SAFE to SYSTEM (non-recoverable → auto-stop
    fires). Severity is NOT inferred from the code's decimal magnitude — the real
    YRC1000 does not encode it that way, and the old scheme dropped codes >=4000 to
    USER/recoverable, suppressing auto-stop on genuine major faults."""

    def test_unknown_code_defaults_to_system(self):
        for code in (1500, 2999, 3500, 4500):
            info = decode_alarm(code)
            assert info.severity == AlarmSeverity.SYSTEM, code
            assert info.name.startswith("UNKNOWN_")
            assert info.code == code

    def test_unknown_high_code_is_not_recoverable(self):
        # The dangerous direction: a >=4000 code must NOT be classified recoverable.
        from src.orchestrator.backends.alarm_codes import is_recoverable
        assert not is_recoverable(decode_alarm(4500))


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
