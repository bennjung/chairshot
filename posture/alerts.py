from __future__ import annotations

import sys
import time
from typing import Optional


class GpioAlert:
    def __init__(
        self,
        green_pin: Optional[int],
        yellow_pin: Optional[int],
        red_pin: Optional[int],
        buzzer_pin: Optional[int],
        switch_delay_seconds: float = 3.0,
    ) -> None:
        self.green = None
        self.yellow = None
        self.red = None
        self.buzzer = None
        self.switch_delay_seconds = switch_delay_seconds
        self.current_led_state: Optional[str] = None
        self.pending_led_state: Optional[str] = None
        self.pending_since: Optional[float] = None
        if green_pin is None and yellow_pin is None and red_pin is None and buzzer_pin is None:
            return
        try:
            from gpiozero import Buzzer, LED
        except ImportError:
            print("gpiozero is not installed; GPIO output disabled.", file=sys.stderr)
            return
        self.green = LED(green_pin) if green_pin is not None else None
        self.yellow = LED(yellow_pin) if yellow_pin is not None else None
        self.red = LED(red_pin) if red_pin is not None else None
        self.buzzer = Buzzer(buzzer_pin) if buzzer_pin is not None else None

    def set_state(self, state: str, alarm: bool) -> None:
        target = self._target_led_state(state)
        self._debounced_apply(target)
        if self.buzzer is not None:
            self.buzzer.on() if alarm else self.buzzer.off()

    def set_baseline(self) -> None:
        self.pending_led_state = None
        self.pending_since = None
        self.current_led_state = "baseline"
        self._apply(green=True, yellow=True, red=True)

    def close(self) -> None:
        for device in (self.green, self.yellow, self.red, self.buzzer):
            if device is not None:
                device.off()
                device.close()

    def _target_led_state(self, state: str) -> str:
        if state == "normal":
            return "normal"
        if state == "out_of_range":
            return "out_of_range"
        if state == "bad":
            return "bad"
        return "out_of_range"

    def _debounced_apply(self, target: str) -> None:
        now = time.monotonic()
        if self.current_led_state is None:
            self._apply_state(target)
            return
        if target == self.current_led_state:
            self.pending_led_state = None
            self.pending_since = None
            return

        if target != self.pending_led_state:
            self.pending_led_state = target
            self.pending_since = now
            return

        if self.pending_since is not None and now - self.pending_since >= self.switch_delay_seconds:
            self._apply_state(target)
            self.pending_led_state = None
            self.pending_since = None

    def _apply_state(self, state: str) -> None:
        self.current_led_state = state
        if state == "normal":
            self._apply(green=True, yellow=False, red=False)
        elif state == "bad":
            self._apply(green=False, yellow=False, red=True)
        elif state == "out_of_range":
            self._apply(green=False, yellow=True, red=False)
        else:
            self._apply(green=False, yellow=False, red=False)

    def _apply(self, green: bool, yellow: bool, red: bool) -> None:
        self._write(self.green, green)
        self._write(self.yellow, yellow)
        self._write(self.red, red)

    @staticmethod
    def _write(device: Optional[object], enabled: bool) -> None:
        if device is None:
            return
        device.on() if enabled else device.off()
