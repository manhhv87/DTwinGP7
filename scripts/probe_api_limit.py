#!/usr/bin/env python
"""
probe_api_limit.py
──────────────────
Đo giới hạn API của RoboDK: gọi liên tục một lệnh API rẻ tiền (ItemList) và
đếm xem chạm giới hạn ("API calls are limited") sau bao nhiêu lệnh.

Mục đích: xác định giới hạn là THEO SỐ LỆNH MỖI PHIÊN hay THEO QUOTA NGÀY.

Cách dùng — chạy 2 LẦN, mỗi lần đều ĐÓNG HẲN RoboDK trước khi chạy:
    # Lần 1: đóng hẳn RoboDK → chạy → ghi lại số N1
    python scripts/probe_api_limit.py
    # Lần 2: đóng hẳn RoboDK lần nữa → chạy → ghi lại số N2
    python scripts/probe_api_limit.py

Đọc kết quả:
    N1 ≈ N2              → giới hạn THEO SỐ LỆNH mỗi phiên (~N lệnh/phiên)
    N2 nhỏ hơn N1 nhiều  → giới hạn THEO QUOTA NGÀY (cộng dồn, không reset khi
                           khởi động lại RoboDK)
    Cả hai chạy hết 1000 → không gặp giới hạn (license đã OK)
"""
from __future__ import annotations

import sys

MAX_CALLS = 1000


def main() -> int:
    try:
        from robodk.robolink import Robolink
    except ImportError:
        print("✗ Chưa cài robodk: pip install robodk")
        return 1

    rdk = Robolink()
    print(f"Bắt đầu gọi API liên tục (tối đa {MAX_CALLS} lệnh)...")

    ok = 0
    try:
        for i in range(1, MAX_CALLS + 1):
            rdk.ItemList()          # 1 lệnh API round-trip, không đổi station
            ok = i
            if i % 25 == 0:
                print(f"  ... {i} lệnh OK")
    except Exception as e:  # noqa: BLE001
        print("=" * 56)
        print(f"✗ CHẠM GIỚI HẠN sau {ok} lệnh API thành công.")
        print(f"  Lỗi: {type(e).__name__}: {e}")
        print("=" * 56)
        print("→ Ghi lại số này. Đóng hẳn RoboDK rồi chạy script lần nữa.")
        return 0

    print("=" * 56)
    print(f"✓ Gọi hết {ok} lệnh API, KHÔNG gặp giới hạn.")
    print("  → License RoboDK đang hoạt động bình thường cho API.")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
