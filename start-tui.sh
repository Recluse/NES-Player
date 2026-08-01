#!/bin/sh
# Text launcher: the menu runs in the terminal
cd "$(dirname "$0")"
exec uv run python scripts/launcher.py
