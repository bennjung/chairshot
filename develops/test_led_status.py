#!/usr/bin/env python3
"""Development-only standalone GPIO LED status test for posture states.

BCM pin defaults:
- green: 17 -> normal
- red: 27 -> turtle_neck / shoulder_tilt / bad posture
- yellow: 22 -> baseline blink, out_of_range solid
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LedPins:
    green: int
    red: int
    yellow: int
    active_high: bool


class LedStatusTester:
    def __init__(self, pins: LedPins, mock: bool = False) -> None:
        self.mock = mock
        self.green = None
        self.red = None
        self.yellow = None
        if mock:
            return

        try:
            from gpiozero import LED
        except ImportError as exc:
            raise SystemExit("gpiozero is not installed. Run: sudo apt install python3-gpiozero") from exc

        self.green = LED(pins.green, active_high=pins.active_high)
        self.red = LED(pins.red, active_high=pins.active_high)
        self.yellow = LED(pins.yellow, active_high=pins.active_high)

    def normal(self) -> None:
        self._set(green=True, red=False, yellow=False, label="normal")

    def bad_posture(self) -> None:
        self._set(green=False, red=True, yellow=False, label="bad posture")

    def baseline_blink(self, duration: float, interval: float) -> None:
        self._set(green=False, red=False, yellow=False, label="baseline blink start")
        end_at = time.monotonic() + duration
        on = False
        while time.monotonic() < end_at:
            on = not on
            self._set(green=False, red=False, yellow=on, label=f"baseline yellow {'on' if on else 'off'}")
            time.sleep(interval)
        self.off()

    def out_of_range(self) -> None:
        self._set(green=False, red=False, yellow=True, label="out_of_range")

    def off(self) -> None:
        self._set(green=False, red=False, yellow=False, label="off")

    def close(self) -> None:
        self.off()
        for device in (self.green, self.red, self.yellow):
            if device is not None:
                device.close()

    def _set(self, green: bool, red: bool, yellow: bool, label: str) -> None:
        if self.mock:
            print(f"[mock] {label}: green={int(green)} red={int(red)} yellow={int(yellow)}", flush=True)
            return
        self._write(self.green, green)
        self._write(self.red, red)
        self._write(self.yellow, yellow)
        print(f"{label}: green={int(green)} red={int(red)} yellow={int(yellow)}", flush=True)

    @staticmethod
    def _write(device: Optional[object], enabled: bool) -> None:
        if device is None:
            return
        device.on() if enabled else device.off()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test green/red/yellow posture status LEDs")
    parser.add_argument("--green-pin", type=int, default=17, help="BCM pin for normal green LED")
    parser.add_argument("--red-pin", type=int, default=27, help="BCM pin for bad posture red LED")
    parser.add_argument("--yellow-pin", type=int, default=22, help="BCM pin for logging/out_of_range yellow LED")
    parser.add_argument("--active-low", action="store_true", help="Use when LEDs turn on with LOW output")
    parser.add_argument("--mock", action="store_true", help="Print states without touching GPIO")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds per steady state")
    parser.add_argument("--blink-interval", type=float, default=0.35, help="Yellow blink interval seconds")
    parser.add_argument(
        "--mode",
        choices=("sequence", "normal", "bad", "baseline", "out-of-range", "off"),
        default="sequence",
    )
    return parser.parse_args()


def run_mode(tester: LedStatusTester, args: argparse.Namespace) -> None:
    if args.mode == "normal":
        tester.normal()
        time.sleep(args.duration)
    elif args.mode == "bad":
        tester.bad_posture()
        time.sleep(args.duration)
    elif args.mode == "baseline":
        tester.baseline_blink(args.duration, args.blink_interval)
    elif args.mode == "out-of-range":
        tester.out_of_range()
        time.sleep(args.duration)
    elif args.mode == "off":
        tester.off()
    else:
        print("1/4 normal: green on", flush=True)
        tester.normal()
        time.sleep(args.duration)

        print("2/4 bad posture: red on", flush=True)
        tester.bad_posture()
        time.sleep(args.duration)

        print("3/4 baseline setting/logging: yellow blink", flush=True)
        tester.baseline_blink(args.duration, args.blink_interval)

        print("4/4 out_of_range: yellow solid", flush=True)
        tester.out_of_range()
        time.sleep(args.duration)


def main() -> None:
    args = parse_args()
    pins = LedPins(
        green=args.green_pin,
        red=args.red_pin,
        yellow=args.yellow_pin,
        active_high=not args.active_low,
    )
    tester = LedStatusTester(pins, mock=args.mock)

    def stop(_signum, _frame) -> None:
        tester.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        run_mode(tester, args)
    finally:
        tester.close()


if __name__ == "__main__":
    main()
