#!/usr/bin/env python3
"""
Require:   pip install pyserial
"""
import base64
import os
import platform
import queue
import re
import sys
import threading
from datetime import datetime

import serial
import serial.tools.list_ports

# ==================== CAU HINH CONG UART ====================
AUTO_DETECT_PORT = True        # True: tu tim cong ESP32/USB-UART; False: dung cong co dinh ben duoi
WIN_PORT = "COM3"              # Cong mac dinh tren Windows (doi neu may ban la COM4, COM5...)
LINUX_PORT = "/dev/ttyUSB0"    # Cong mac dinh tren Linux (/dev/ttyUSB0 hoac /dev/ttyACM0)
BAUD_RATE = 115200
OUT_DIR = "captured_images"
SHOW_PREVIEW = True            # Hien cua so xem lai anh vua chup (neu co opencv-python)
# ============================================================

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import msvcrt
else:
    import termios
    import tty


def find_serial_port() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    if AUTO_DETECT_PORT:
        ports = list(serial.tools.list_ports.comports())
        keywords = ("cp210", "ch340", "ch341", "ftdi", "usb", "jtag", "esp", "uart", "acm")
        for p in ports:
            desc = f"{p.description} {p.manufacturer or ''} {p.device}".lower()
            if any(k in desc for k in keywords):
                return p.device
        if ports:
            return ports[0].device

    return WIN_PORT if IS_WINDOWS else LINUX_PORT


def uart_reader(
    ser: serial.Serial,
    out_dir: str,
    stop_event: threading.Event,
    preview_q: "queue.Queue[bytes]",
):
    receiving = False
    b64_chunks = []
    count = 0

    while not stop_event.is_set():
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()

        if line.startswith("<<<IMG_START"):
            receiving = True
            b64_chunks = []
            m = re.search(r"len=(\d+)", line)
            expected = m.group(1) if m else "?"
            print(f"\nDang nhan anh ({expected} bytes JPEG)...")
            continue

        if line == "<<<IMG_END>>>" and receiving:
            receiving = False
            try:
                jpg_bytes = base64.b64decode("".join(b64_chunks))
                count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"img_{ts}_{count:04d}.jpg")
                with open(path, "wb") as f:
                    f.write(jpg_bytes)
                print(f"Da luu: {path} ({len(jpg_bytes)} bytes)")
                if SHOW_PREVIEW and HAS_CV2:
                    preview_q.put(jpg_bytes)
            except Exception as e:
                print(f"Loi giai ma anh: {e}")
            continue

        if receiving:
            b64_chunks.append(line)
        elif line:
            print(f"[ESP32] {line}")


def key_listener(ser: serial.Serial, stop_event: threading.Event):
    while not stop_event.is_set():
        if IS_WINDOWS:
            if not msvcrt.kbhit():
                stop_event.wait(0.02)
                continue
            ch = msvcrt.getwch()
        else:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        if ch in ("c", "C"):
            ser.write(b"c")
            ser.flush()
        elif ch in ("q", "Q", "\x03"):
            stop_event.set()
            break


def main():
    port = find_serial_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else BAUD_RATE
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Dang mo UART tai {port} ({baud} baud)...")
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
    except serial.SerialException as e:
        print(f"Khong mo duoc cong {port}: {e}")
        print("Hay kiem tra lai WIN_PORT / LINUX_PORT trong file nay hoac tat Serial Monitor.")
        return

    stop_event = threading.Event()
    preview_q: "queue.Queue[bytes]" = queue.Queue()

    t_read = threading.Thread(target=uart_reader, args=(ser, OUT_DIR, stop_event, preview_q), daemon=True)
    t_key = threading.Thread(target=key_listener, args=(ser, stop_event), daemon=True)
    t_read.start()
    t_key.start()

    print("San sang! Nhan phim 'c' / 'C' de chup anh, nhan 'q' de thoat.")
    if SHOW_PREVIEW and not HAS_CV2:
        print("(Meo: cai 'pip install opencv-python' neu muon tu dong mo cua so xem anh vua chup)")

    try:
        while not stop_event.is_set():
            if SHOW_PREVIEW and HAS_CV2:
                try:
                    jpg_bytes = preview_q.get(timeout=0.05)
                    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imshow("OV7670 Capture", img)
                except queue.Empty:
                    pass
                cv2.waitKey(1)
            else:
                stop_event.wait(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        t_read.join(timeout=1)
        ser.close()
        if HAS_CV2:
            cv2.destroyAllWindows()
        print("\nDa dong UART.")


if __name__ == "__main__":
    main()
