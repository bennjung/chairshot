from __future__ import annotations

from .models import PostureResult


POSTURE_STATES = ("normal", "turtle_neck", "shoulder_tilt", "out_of_range")


def canonical_posture_state(result: PostureResult) -> str:
    if result.reason in ("turtle_neck", "shoulder_tilt"):
        return result.reason
    if result.state == "normal" or result.reason == "normal":
        return "normal"
    return "out_of_range"
