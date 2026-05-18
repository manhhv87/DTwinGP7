"""
exceptions.py
─────────────
Custom exceptions cho cell management module.

Tách riêng giúp:
  - User code có thể catch specific exceptions
  - Error messages có cấu trúc, rõ ràng
  - Dễ test với pytest.raises(SpecificError)
"""
from __future__ import annotations


class CellError(Exception):
    """Base exception cho mọi lỗi liên quan đến cell management."""


class RoboDKConnectionError(CellError):
    """Không kết nối được tới RoboDK Software.

    Nguyên nhân thường gặp:
        - RoboDK GUI chưa mở
        - RoboDK bị treo
        - Port 20500 bị block
    """


class InvalidConfigError(CellError):
    """File YAML config không hợp lệ.

    Bao gồm:
        - YAML syntax error
        - Schema validation failure (Pydantic)
        - Logical inconsistency (vd parent_frame không tồn tại)
    """


class MissingMeshError(CellError):
    """File mesh (STL/OBJ/...) được tham chiếu trong config nhưng không tồn tại."""

    def __init__(self, mesh_path: str, item_name: str | None = None):
        msg = f"Mesh file not found: {mesh_path}"
        if item_name:
            msg += f" (referenced by '{item_name}')"
        super().__init__(msg)
        self.mesh_path = mesh_path
        self.item_name = item_name


class MissingRobotError(CellError):
    """Robot không tìm thấy trong RoboDK Library.

    User cần drag robot từ Online Library trước khi chạy script.
    """

    def __init__(self, robot_name: str, library_path: str | None = None):
        msg = (
            f"Robot '{robot_name}' không có trong Library.\n"
            f"Cách fix:\n"
            f"  1. Mở RoboDK GUI\n"
            f"  2. File > Open Online Library\n"
            f"  3. Search '{robot_name}'\n"
            f"  4. Drag vào station → file sẽ tự download về"
        )
        if library_path:
            msg += f"\n  Library path: {library_path}"
        super().__init__(msg)
        self.robot_name = robot_name


class RobotConnectionError(CellError):
    """Không kết nối được tới robot thật qua RoboDK driver.

    Nguyên nhân:
        - IP sai
        - YRC1000 chưa bật High Speed Ethernet Server
        - Firewall block
        - Cable Ethernet lỏng
    """

    def __init__(self, ip: str, port: int, driver: str):
        msg = (
            f"Không kết nối được tới robot tại {ip}:{port} (driver: {driver}).\n"
            f"Kiểm tra:\n"
            f"  - Ping {ip} có thành công không\n"
            f"  - High Speed Ethernet Server đã enable trên YRC1000\n"
            f"  - Firewall Windows không block RoboDK\n"
            f"  - PC và YRC1000 cùng subnet"
        )
        super().__init__(msg)
        self.ip = ip
        self.port = port
        self.driver = driver
