"""
inform_codegen.py
─────────────────
Sinh job file INFORM (.JBI) cho Yaskawa YRC1000.

INFORM là ngôn ngữ lập trình robot riêng của Yaskawa. Mỗi file .JBI là 1 chương
trình mà robot chạy khi nhận JOB_SELECT + START qua HSE.

Format reference:
  - Yaskawa "INFORM Language Manual" (RE-CKI-A464)
  - C-variables (C00000-C99999): position constants
  - POSTYPE PULSE: raw encoder pulses (đơn vị 0.0001° trên axis nào tuỳ ratio)
  - Lệnh chính: MOVJ (joint), MOVL (linear), DOUT (digital output), TIMER

Module này PURE TEXT — không IO. Test bằng snapshot string. Upload qua FTP
ở `motoman_hse.py`.

Workflow:
  builder = InformJobBuilder("PICKPLACE")
  builder.add_position("p0", joints_deg=[0,0,0,0,0,0])
  builder.add_position("p1", joints_deg=[10,-5,20,0,30,-15])
  builder.movj("p0", speed_pct=10.0)
  builder.movj("p1", speed_pct=10.0)
  builder.dout(1, True)        # gripper close
  builder.timer(0.3)
  builder.movj("p0", speed_pct=10.0)
  builder.dout(1, False)
  text = builder.render()       # str → ghi .JBI rồi FTP upload
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .hse_protocol import GP7_PULSE_PER_DEG

# Max 999 C-variables/job nhưng giữ ngưỡng thấp cho safety.
MAX_POSITIONS_PER_JOB = 100

# Speed limits (Yaskawa convention) — clamp ở đây để tránh user truyền 100%.
MAX_SPEED_PCT_DEFAULT = 30.0     # joint speed %
MAX_LINEAR_MM_S = 250.0          # cartesian mm/s

PosType = Literal["PULSE", "BASE", "ROBOT", "USER"]


# ───── Internal instruction dataclasses ─────


@dataclass
class _Position:
    """C-variable position constant."""

    name: str
    joints_pulse: list[int]
    tool_no: int = 0
    user_frame: int = 0


@dataclass
class _Instruction:
    """1 dòng INFORM trong section //INST."""

    text: str


# ───── Builder ─────


class InformJobBuilder:
    """Build INFORM .JBI từ python-level commands.

    Args:
        name: Tên job (≤ 32 ASCII chars, không space).
        pos_type: POSTYPE — PULSE (default) cho joint pulses.
        pulse_per_deg: Ratio pulse/degree cho từng axis. Default GP7.
        max_speed_pct: Cap tốc độ joint cho safety. Default 30%.
        group: GROUP1 RB1 (single robot). Đổi nếu multi-robot.
    """

    def __init__(
        self,
        name: str,
        pos_type: PosType = "PULSE",
        pulse_per_deg: tuple[float, ...] = GP7_PULSE_PER_DEG,
        max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
        group: str = "RB1",
    ) -> None:
        self._validate_name(name)
        self.name = name
        self.pos_type = pos_type
        self.pulse_per_deg = pulse_per_deg
        self.max_speed_pct = max_speed_pct
        self.group = group
        self._positions: list[_Position] = []
        self._pos_index: dict[str, int] = {}      # name → idx trong _positions
        self._instructions: list[_Instruction] = []

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or len(name) > 32:
            raise ValueError(f"Job name phải 1-32 ký tự, nhận '{name}'")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Job name chỉ chữ/số/_, nhận '{name}'")

    # ─── Positions ───
    def add_position(self, name: str, joints_deg: list[float]) -> "InformJobBuilder":
        """Thêm 1 C-variable position từ joints (degrees)."""
        if name in self._pos_index:
            raise ValueError(f"Position '{name}' đã tồn tại")
        if len(joints_deg) != len(self.pulse_per_deg):
            raise ValueError(
                f"Joints phải {len(self.pulse_per_deg)} phần tử, "
                f"nhận {len(joints_deg)}"
            )
        if len(self._positions) >= MAX_POSITIONS_PER_JOB:
            raise ValueError(
                f"Đã đạt giới hạn {MAX_POSITIONS_PER_JOB} positions/job. "
                f"Tách job lớn thành nhiều file."
            )
        pulses = [int(round(d * r)) for d, r in zip(joints_deg, self.pulse_per_deg)]
        self._pos_index[name] = len(self._positions)
        self._positions.append(_Position(name=name, joints_pulse=pulses))
        return self

    def _resolve_cvar(self, position_name: str) -> str:
        """Map logical position name → C-variable token (vd 'p0' → 'C00000')."""
        if position_name not in self._pos_index:
            raise KeyError(f"Position '{position_name}' chưa add — gọi add_position trước")
        return f"C{self._pos_index[position_name]:05d}"

    # ─── Motion instructions ───
    def movj(self, position_name: str, speed_pct: float | None = None,
             tool_no: int | None = None) -> "InformJobBuilder":
        """Joint move tới position (cao tốc, không quan tâm path).

        Args:
            tool_no: Override TOOL coordinate. None = không thêm TL= (dùng
                TOOL hiện tại). Đặt 1 cho TOOL01 (gripper).
        """
        cvar = self._resolve_cvar(position_name)
        vj = self._clamp_joint_speed(speed_pct)
        inst = f"MOVJ {cvar} VJ={vj:.2f}"
        if tool_no is not None:
            inst += f" TL={tool_no}"
        self._instructions.append(_Instruction(inst))
        return self

    def movl(
        self,
        position_name: str,
        speed_mm_s: float = 100.0,
        tool_no: int | None = None,
    ) -> "InformJobBuilder":
        """Linear (Cartesian) move."""
        cvar = self._resolve_cvar(position_name)
        v = max(1.0, min(float(speed_mm_s), MAX_LINEAR_MM_S))
        inst = f"MOVL {cvar} V={v:.1f}"
        if tool_no is not None:
            inst += f" TL={tool_no}"
        self._instructions.append(_Instruction(inst))
        return self

    # ─── I/O + timing ───
    def dout(self, output_index: int, on: bool) -> "InformJobBuilder":
        """Digital output ON/OFF — điều khiển gripper / signal lamp."""
        if not (1 <= output_index <= 1024):
            raise ValueError(f"DOUT index ngoài range 1..1024: {output_index}")
        state = "ON" if on else "OFF"
        self._instructions.append(_Instruction(f"DOUT OT#({output_index}) {state}"))
        return self

    def timer(self, seconds: float) -> "InformJobBuilder":
        """Pause `seconds` giây (gripper settle time, etc)."""
        if seconds < 0 or seconds > 600:
            raise ValueError(f"TIMER ngoài range 0-600s: {seconds}")
        self._instructions.append(_Instruction(f"TIMER T={seconds:.2f}"))
        return self

    def comment(self, text: str) -> "InformJobBuilder":
        """Thêm comment 'NOP // <text>' để debug."""
        safe = text.replace("\r", "").replace("\n", " ")[:80]
        self._instructions.append(_Instruction(f"'{safe}"))    # ' = INFORM comment
        return self

    def _clamp_joint_speed(self, speed_pct: float | None) -> float:
        if speed_pct is None:
            return self.max_speed_pct
        return max(0.01, min(float(speed_pct), self.max_speed_pct))

    # ─── Render ───
    def render(self, date_str: str = "2026/01/01 00:00") -> str:
        """Sinh full INFORM .JBI text. CRLF line ending (Yaskawa convention)."""
        if not self._positions:
            raise ValueError("Job phải có ít nhất 1 position")
        if not self._instructions:
            raise ValueError("Job phải có ít nhất 1 instruction")

        lines: list[str] = []
        lines.append("/JOB")
        lines.append(f"//NAME {self.name}")
        lines.append("//POS")
        # NPOS = số positions cho mỗi group (BP/EX/ST/EXP = 0 cho single robot 6-axis).
        lines.append(f"///NPOS {len(self._positions)},0,0,0,0,0")
        lines.append(f"///TOOL {self._positions[0].tool_no}")
        lines.append(f"///POSTYPE {self.pos_type}")
        # Section đầu (PULSE/BASE/...) khớp pos_type.
        lines.append(f"///{self.pos_type}")
        for p in self._positions:
            cvar = f"C{self._pos_index[p.name]:05d}"
            values = ",".join(str(v) for v in p.joints_pulse)
            lines.append(f"{cvar}={values}")

        lines.append("//INST")
        lines.append(f"///DATE {date_str}")
        lines.append("///ATTR SC,RW")
        lines.append(f"///GROUP1 {self.group}")
        lines.append("NOP")
        for inst in self._instructions:
            lines.append(inst.text)
        lines.append("END")

        return "\r\n".join(lines) + "\r\n"


# ───── Ultra-fast P-variable template (M3++) ─────


def gen_pvar_template_job(
    name: str,
    num_positions: int,
    motion_kinds: list[str] | None = None,
    gripper_do_index: int = 1,
    gripper_close_at: int | None = None,
    gripper_open_at: int | None = None,
    gripper_delay_s: float = 0.3,
    max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
) -> str:
    """Sinh INFORM template dùng P-variables runtime-mutable.

    Template upload 1 lần đầu, sau đó mỗi trial chỉ cần WRITE_POS_VAR + START.
    Tốc độ siêu nhanh (~50ms/trial overhead) vì không có FTP roundtrip.

    P-variables KHÔNG khai báo trong //POS section — chúng là runtime vars
    set qua HSE WRITE_POS_VAR. Job tham chiếu P000, P001, ... trong //INST.

    Args:
        name: Tên job (≤ 32 ASCII).
        num_positions: Số P-variables job MOVJ/MOVL tới (= số waypoint).
        motion_kinds: List "movj"/"movl" cho mỗi position. Default all movj.
        gripper_close_at: Index để chèn DOUT ON trước MOVJ đó. None = không.
        gripper_open_at: Index để chèn DOUT OFF trước MOVJ đó.
        gripper_delay_s: TIMER giây sau mỗi DOUT (gripper settle).
        max_speed_pct: Speed cap cho VJ.

    Returns:
        INFORM .JBI text dùng P-variables.
    """
    if num_positions < 1 or num_positions > 128:
        raise ValueError(f"num_positions 1-128, got {num_positions}")
    if motion_kinds is None:
        motion_kinds = ["movj"] * num_positions
    if len(motion_kinds) != num_positions:
        raise ValueError(
            f"motion_kinds phải có {num_positions} entries, got {len(motion_kinds)}"
        )

    vj = float(max_speed_pct)
    lines: list[str] = []
    lines.append("/JOB")
    lines.append(f"//NAME {name}")
    lines.append("//POS")
    lines.append("///NPOS 0,0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")
    lines.append("//INST")
    lines.append("///DATE 2026/01/01 00:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.append("'P-var template - runtime set via HSE WRITE_POS_VAR")

    for i in range(num_positions):
        if gripper_close_at is not None and i == gripper_close_at:
            lines.append(f"DOUT OT#({gripper_do_index}) ON")
            lines.append(f"TIMER T={gripper_delay_s:.2f}")
        if gripper_open_at is not None and i == gripper_open_at:
            lines.append(f"DOUT OT#({gripper_do_index}) OFF")
            lines.append(f"TIMER T={gripper_delay_s:.2f}")

        kind = motion_kinds[i].upper()
        if kind == "MOVJ":
            lines.append(f"MOVJ P{i:03d} VJ={vj:.2f}")
        elif kind == "MOVL":
            lines.append(f"MOVL P{i:03d} V=80.0")
        else:
            raise ValueError(f"motion_kinds[{i}]='{motion_kinds[i]}' không hỗ trợ")

    lines.append("END")
    return "\r\n".join(lines) + "\r\n"


# ───── High-level convenience ─────


def gen_pick_place_job(
    name: str,
    home_deg: list[float],
    approach_deg: list[float],
    grasp_deg: list[float],
    transfer_deg: list[float],
    place_deg: list[float],
    gripper_do_index: int = 1,
    gripper_delay_s: float = 0.3,
    speed_pct: float = 10.0,
    max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
) -> str:
    """Sinh INFORM .JBI cho 1 chu trình pick-and-place đầy đủ.

    Workflow: home → approach → grasp (đóng + nhấc) → transfer → place
    (hạ + mở) → home. Tốc độ luôn ≤ max_speed_pct.
    """
    builder = (
        InformJobBuilder(name=name, max_speed_pct=max_speed_pct)
        .add_position("home", home_deg)
        .add_position("approach", approach_deg)
        .add_position("grasp", grasp_deg)
        .add_position("transfer", transfer_deg)
        .add_position("place", place_deg)
        .comment("Pick-and-place sinh tự động từ Orchestrator")
        .movj("home", speed_pct=speed_pct)
        .movj("approach", speed_pct=speed_pct)
        .movj("grasp", speed_pct=speed_pct)
        .dout(gripper_do_index, True)             # đóng gripper
        .timer(gripper_delay_s)
        .movj("approach", speed_pct=speed_pct)    # nhấc lên
        .movj("transfer", speed_pct=speed_pct)
        .movj("place", speed_pct=speed_pct)
        .dout(gripper_do_index, False)            # mở gripper
        .timer(gripper_delay_s)
        .movj("transfer", speed_pct=speed_pct)    # nhấc lên
        .movj("home", speed_pct=speed_pct)
    )
    return builder.render()
