#!/bin/sh
# Graphical launcher; the menu returns after the game exits
cd "$(dirname "$0")"
exec uv run python scripts/launcher_gui.py
