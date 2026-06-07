#!/usr/bin/env python3
"""Development-only posture LED status test.

BCM pin defaults from ledtest.py:
- GPIO 18: green  -> normal
- GPIO 23: yellow -> baseline setting / out_of_range
- GPIO 24: red    -> bad posture alarm
"""

from __future__ import annotations

import argparse
import time
from time import sleep

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


PREVIEW_ENABLED = False
PREVIEW_WINDOW = "Posture LED Test"


def set_preview_enabled(enabled: bool) -> None:
    global PREVIEW_ENABLED
    PREVIEW_ENABLED = enabled
    if enabled and (cv2 is None or np is None):
        raise SystemExit("OpenCV/numpy is not installed. Install python3-opencv and python3-numpy.")


def show_preview(state: str, led_text: str, color: tuple[int, int, int]) -> None:
    if not PREVIEW_ENABLED:
        return
    assert cv2 is not None
    assert np is not None

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (18, 24, 30)

    cv2.rectangle(image, (24, 24), (616, 336), (45, 58, 70), 2)
    cv2.putText(image, "POSTURE LED TEST", (42, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 230, 240), 2, cv2.LINE_AA)
    cv2.putText(image, f"STATE: {state}", (42, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.putText(image, led_text, (42, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(image, "press q or ESC to close preview", (42, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 165, 175), 2, cv2.LINE_AA)

    cv2.imshow(PREVIEW_WINDOW, image)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        close_preview()


def preview_sleep(seconds: float) -> None:
    if not PREVIEW_ENABLED:
        sleep(seconds)
        return
    end_at = time.monotonic() + seconds
    while time.monotonic() < end_at:
        assert cv2 is not None
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            close_preview()
            return


def close_preview() -> None:
    if PREVIEW_ENABLED and cv2 is not None:
        cv2.destroyWindow(PREVIEW_WINDOW)


class PostureLedTester:
    def __init__(self, green_pin: int, yellow_pin: int, red_pin: int) -> None:
        try:
            from gpiozero import LED
        except ImportError as exc:
            raise SystemExit("gpiozero is not installed. Run: sudo apt install python3-gpiozero") from exc

        self.green = LED(green_pin)
        self.yellow = LED(yellow_pin)
        self.red = LED(red_pin)

    def led_on(self) -> None:
        self.green.on()
        self.yellow.on()
        self.red.on()

    def led_off(self) -> None:
        self.green.off()
        self.yellow.off()
        self.red.off()

    def baseline_setting(self) -> None:
        self.led_on()
        print("baseline_setting: green=1 yellow=1 red=1", flush=True)
        show_preview("baseline_setting", "ALL LEDs ON", (0, 255, 255))

    def normal(self) -> None:
        self.led_off()
        self.green.on()
        print("normal: green=1 yellow=0 red=0", flush=True)
        show_preview("normal", "GREEN ON", (0, 255, 0))

    def bad_posture_alarm(self) -> None:
        self.led_off()
        self.red.on()
        print("bad_posture_alarm: green=0 yellow=0 red=1", flush=True)
        show_preview("bad_posture_alarm", "RED ON", (0, 0, 255))

    def out_of_range(self) -> None:
        self.led_off()
        self.yellow.on()
        print("out_of_range: green=0 yellow=1 red=0", flush=True)
        show_preview("out_of_range", "YELLOW ON", (0, 255, 255))

    def close(self) -> None:
        self.led_off()
        self.green.close()
        self.yellow.close()
        self.red.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test posture LED status mapping")
    parser.add_argument("--green-pin", type=int, default=18)
    parser.add_argument("--yellow-pin", type=int, default=23)
    parser.add_argument("--red-pin", type=int, default=24)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--preview", action="store_true", help="Show OpenCV status preview window")
    parser.add_argument(
        "--mode",
        choices=("sequence", "baseline", "normal", "bad", "out-of-range", "off"),
        default="sequence",
    )
    return parser.parse_args()


def run_mode(tester: PostureLedTester, mode: str, duration: float) -> None:
    if mode == "baseline":
        tester.baseline_setting()
        preview_sleep(duration)
    elif mode == "normal":
        tester.normal()
        preview_sleep(duration)
    elif mode == "bad":
        tester.bad_posture_alarm()
        preview_sleep(duration)
    elif mode == "out-of-range":
        tester.out_of_range()
        preview_sleep(duration)
    elif mode == "off":
        tester.led_off()
    else:
        print("1/4 baseline setting: all LEDs on", flush=True)
        tester.baseline_setting()
        preview_sleep(duration)

        print("2/4 normal: green LED on", flush=True)
        tester.normal()
        preview_sleep(duration)

        print("3/4 bad posture over 5s: red LED on", flush=True)
        tester.bad_posture_alarm()
        preview_sleep(duration)

        print("4/4 out_of_range: yellow LED on", flush=True)
        tester.out_of_range()
        preview_sleep(duration)


def main() -> None:
    args = parse_args()
    set_preview_enabled(args.preview)
    tester = PostureLedTester(args.green_pin, args.yellow_pin, args.red_pin)
    try:
        run_mode(tester, args.mode, args.duration)
    finally:
        tester.close()
        close_preview()


if __name__ == "__main__":
    main()
