"""
alarm_codes.py
──────────────
Yaskawa YRC1000 alarm code decoder.

YRC1000 classifies alarms into 4 levels: MINOR (1xxx), MAJOR (2xxx), SYSTEM (3xxx),
USER (4xxx). HSE READ_ALARM (0x70) returns (code, sub_code, severity) — this module
translates them into human-readable strings + severity hints.

The alarm database below is ABBREVIATED — the Yaskawa "Alarm List Manual" contains
hundreds of codes. Add entries as they are encountered at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class AlarmSeverity(IntEnum):
    """Alarm severity level — determines the recovery strategy."""

    NONE = 0           # No active alarm
    MINOR = 1          # Resettable via HSE/TP
    MAJOR = 2          # Requires manual intervention
    SYSTEM = 3         # Requires controller restart
    USER = 4           # User code error (bad INFORM job)


@dataclass(frozen=True)
class AlarmInfo:
    code: int
    name: str
    severity: AlarmSeverity
    description: str
    recovery_hint: str


# Common alarm codes (verify against Yaskawa "Alarm List Manual" YRC1000).
# Format: code → AlarmInfo. Sub-code (axis number, etc.) is handled separately.
_ALARM_DB: dict[int, AlarmInfo] = {
    0: AlarmInfo(0, "NONE", AlarmSeverity.NONE, "No active alarm",
                 "Continue normally"),

    # 1xxx — MINOR
    1010: AlarmInfo(1010, "REMOTE_MODE_REQUIRED", AlarmSeverity.MINOR,
                    "Remote command received but controller is not in REMOTE mode",
                    "Enable REMOTE on the teach pendant"),
    1020: AlarmInfo(1020, "OVER_RUN", AlarmSeverity.MINOR,
                    "Joint exceeded soft limit",
                    "Manually jog robot out of the restricted zone + reset"),

    # 2xxx — MAJOR
    2010: AlarmInfo(2010, "EMERGENCY_STOP", AlarmSeverity.MAJOR,
                    "E-stop button or safety circuit triggered",
                    "Reset E-stop physically + re-enable servo on TP"),
    2020: AlarmInfo(2020, "SERVO_OFF", AlarmSeverity.MAJOR,
                    "Servo disabled (manual or triggered by another alarm)",
                    "Enable servo via HSE HOLD_SERVO=1 or TP"),
    2030: AlarmInfo(2030, "OVERLOAD", AlarmSeverity.MAJOR,
                    "Motor overload (incorrect CoM payload or collision)",
                    "Check payload, gripper, inspect for mechanical collision"),
    2040: AlarmInfo(2040, "POSITION_ERROR", AlarmSeverity.MAJOR,
                    "Position deviation between command and encoder is too large",
                    "Re-home robot, check encoder cable"),

    # 3xxx — SYSTEM
    3010: AlarmInfo(3010, "ENCODER_BATTERY_LOW", AlarmSeverity.SYSTEM,
                    "Encoder battery weak — zero position will be lost on power-off",
                    "Replace encoder battery, re-zero robot"),
    3020: AlarmInfo(3020, "AMP_FAULT", AlarmSeverity.SYSTEM,
                    "Servo amplifier fault",
                    "Restart controller; if fault repeats, call service"),

    # 4xxx — USER (INFORM job errors)
    4010: AlarmInfo(4010, "JOB_NOT_FOUND", AlarmSeverity.USER,
                    "JOB_SELECT could not find the specified job",
                    "Verify FTP upload succeeded + job name is correct"),
    4020: AlarmInfo(4020, "INFORM_SYNTAX", AlarmSeverity.USER,
                    "Syntax error in .JBI file",
                    "Re-gen INFORM with InformJobBuilder + verify CRLF + ASCII"),
    4030: AlarmInfo(4030, "POSITION_OUT_OF_RANGE", AlarmSeverity.USER,
                    "C-variable position exceeds joint soft limit",
                    "Clamp target joints in reachability check before generating job"),
}


def decode_alarm(code: int, sub_code: int = 0) -> AlarmInfo:
    """Lookup alarm code → AlarmInfo. Sub-code carries axis number or detail.

    Returns:
        AlarmInfo. If code is not in DB → unknown alarm with severity SYSTEM.
    """
    info = _ALARM_DB.get(code)
    if info is not None:
        return info
    # Unknown code → FAIL-SAFE. The real YRC1000 does NOT encode severity in the
    # code's decimal magnitude, so do NOT infer it (the old range scheme dropped
    # every code >=4000 to USER/recoverable, suppressing the twin's auto-stop on
    # genuine major faults such as 41xx servo/encoder). Default any unknown nonzero
    # code to SYSTEM (non-recoverable) so auto-stop fires — matches this function's
    # documented contract.
    severity = AlarmSeverity.NONE if code == 0 else AlarmSeverity.SYSTEM
    return AlarmInfo(
        code=code,
        name=f"UNKNOWN_{code:04d}",
        severity=severity,
        description=f"Alarm code {code} (sub {sub_code}) not found in DB",
        recovery_hint="Look up 'Alarm List Manual' YRC1000 + add entry to alarm_codes.py",
    )


def is_recoverable(info: AlarmInfo) -> bool:
    """Returns True if the alarm can be reset via HSE/script without manual intervention."""
    return info.severity in (AlarmSeverity.NONE, AlarmSeverity.MINOR, AlarmSeverity.USER)
