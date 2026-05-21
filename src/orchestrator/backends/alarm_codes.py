"""
alarm_codes.py
──────────────
Yaskawa YRC1000 alarm code decoder.

YRC1000 phân alarm thành 4 cấp: MINOR (1xxx), MAJOR (2xxx), SYSTEM (3xxx),
USER (4xxx). HSE READ_ALARM (0x70) trả về (code, sub_code, severity) — module
này dịch sang human-readable string + severity hint.

Database alarm dưới đây là RÚT GỌN — Yaskawa "Alarm List Manual" có hàng
trăm code. Thêm khi gặp ở runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class AlarmSeverity(IntEnum):
    """Mức độ nghiêm trọng — quyết định strategy recovery."""

    NONE = 0           # Không có alarm
    MINOR = 1          # Có thể reset qua HSE/TP
    MAJOR = 2          # Cần manual intervention
    SYSTEM = 3         # Cần restart controller
    USER = 4           # Lỗi user code (INFORM job sai)


@dataclass(frozen=True)
class AlarmInfo:
    code: int
    name: str
    severity: AlarmSeverity
    description: str
    recovery_hint: str


# Database alarm codes thông dụng (verify với "Alarm List Manual" YRC1000).
# Format: code → AlarmInfo. Sub-code (axis number etc.) handle riêng.
_ALARM_DB: dict[int, AlarmInfo] = {
    0: AlarmInfo(0, "NONE", AlarmSeverity.NONE, "Không có alarm",
                 "Tiếp tục bình thường"),

    # 1xxx — MINOR
    1010: AlarmInfo(1010, "REMOTE_MODE_REQUIRED", AlarmSeverity.MINOR,
                    "Lệnh từ xa nhưng controller không ở REMOTE mode",
                    "Bật REMOTE trên teach pendant"),
    1020: AlarmInfo(1020, "OVER_RUN", AlarmSeverity.MINOR,
                    "Joint vượt soft limit",
                    "Manual jog robot ra khỏi vùng cấm + reset"),

    # 2xxx — MAJOR
    2010: AlarmInfo(2010, "EMERGENCY_STOP", AlarmSeverity.MAJOR,
                    "E-stop button hoặc safety circuit kích hoạt",
                    "Reset E-stop vật lý + bật lại servo trên TP"),
    2020: AlarmInfo(2020, "SERVO_OFF", AlarmSeverity.MAJOR,
                    "Servo bị tắt (manual hoặc do alarm khác)",
                    "Bật servo qua HSE HOLD_SERVO=1 hoặc TP"),
    2030: AlarmInfo(2030, "OVERLOAD", AlarmSeverity.MAJOR,
                    "Motor quá tải (CoM payload sai hoặc va chạm)",
                    "Check payload, gripper, kiểm tra va chạm cơ học"),
    2040: AlarmInfo(2040, "POSITION_ERROR", AlarmSeverity.MAJOR,
                    "Lệch vị trí command vs encoder quá lớn",
                    "Re-home robot, check encoder cable"),

    # 3xxx — SYSTEM
    3010: AlarmInfo(3010, "ENCODER_BATTERY_LOW", AlarmSeverity.SYSTEM,
                    "Pin encoder yếu — sẽ mất zero position khi power off",
                    "Thay pin encoder, re-zero robot"),
    3020: AlarmInfo(3020, "AMP_FAULT", AlarmSeverity.SYSTEM,
                    "Lỗi servo amplifier",
                    "Restart controller, nếu lặp lại thì service"),

    # 4xxx — USER (INFORM job errors)
    4010: AlarmInfo(4010, "JOB_NOT_FOUND", AlarmSeverity.USER,
                    "JOB_SELECT không tìm thấy job đã chỉ định",
                    "Verify FTP upload thành công + tên job đúng"),
    4020: AlarmInfo(4020, "INFORM_SYNTAX", AlarmSeverity.USER,
                    "Syntax error trong .JBI file",
                    "Re-gen INFORM với InformJobBuilder + verify CRLF + ASCII"),
    4030: AlarmInfo(4030, "POSITION_OUT_OF_RANGE", AlarmSeverity.USER,
                    "C-variable position vượt joint soft limit",
                    "Clamp target joints trong reachability check trước khi gen job"),
}


def decode_alarm(code: int, sub_code: int = 0) -> AlarmInfo:
    """Lookup alarm code → AlarmInfo. Sub-code chứa axis number hoặc detail.

    Returns:
        AlarmInfo. Nếu code không trong DB → unknown alarm với severity SYSTEM.
    """
    info = _ALARM_DB.get(code)
    if info is not None:
        return info
    # Unknown code — guess severity từ range
    if code == 0:
        severity = AlarmSeverity.NONE
    elif code < 2000:
        severity = AlarmSeverity.MINOR
    elif code < 3000:
        severity = AlarmSeverity.MAJOR
    elif code < 4000:
        severity = AlarmSeverity.SYSTEM
    else:
        severity = AlarmSeverity.USER
    return AlarmInfo(
        code=code,
        name=f"UNKNOWN_{code:04d}",
        severity=severity,
        description=f"Alarm code {code} (sub {sub_code}) chưa có trong DB",
        recovery_hint="Tra cứu 'Alarm List Manual' YRC1000 + thêm vào alarm_codes.py",
    )


def is_recoverable(info: AlarmInfo) -> bool:
    """Có thể reset alarm qua HSE/script không, hay cần manual intervention?"""
    return info.severity in (AlarmSeverity.NONE, AlarmSeverity.MINOR, AlarmSeverity.USER)
