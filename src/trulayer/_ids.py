from __future__ import annotations

import os
import time


def new_id() -> str:
    """Generate a UUIDv7 string using stdlib only (Python 3.11+ compatible).

    UUIDv7 layout (RFC 9562):
      bits 127-80 (48 bits): Unix epoch in milliseconds
      bits 79-76  (4 bits):  version = 7
      bits 75-64  (12 bits): random (rand_a)
      bits 63-62  (2 bits):  variant = 0b10
      bits 61-0   (62 bits): random (rand_b)
    """
    ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF  # 48-bit ms timestamp
    rand = int.from_bytes(os.urandom(10), "big")  # 80 random bits
    rand_a = (rand >> 68) & 0xFFF  # top 12 bits
    rand_b = rand & 0x3FFFFFFFFFFFFFFF  # bottom 62 bits

    uuid_int = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{uuid_int:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
