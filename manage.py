#!/usr/bin/env python3
"""Compatibility launcher for the Flask posture dashboard server."""

from __future__ import annotations

import sys

from flask_server import main as flask_main


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "runserver":
        host = "0.0.0.0"
        port = "8800"
        rest = []
        for arg in sys.argv[2:]:
            if ":" in arg:
                host, port = arg.rsplit(":", 1)
            elif arg.isdigit():
                port = arg
            else:
                rest.append(arg)
        sys.argv = [sys.argv[0], "--host", host, "--port", port, *rest]
    flask_main()


if __name__ == "__main__":
    main()
