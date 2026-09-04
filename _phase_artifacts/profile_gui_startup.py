"""Phase 12 - Measure GUI startup latency with background detector init.

Measures the user experience:
  T0: process start
  T1: Tk window created
  T2: _init_detector called (launches background thread)
  T3: detector ready (self._detector assigned)

Key metric: T0 to T1 (window appears) vs T0 to T3 (model ready).
BEFORE: both ~2.7s (window frozen).
AFTER:  T0-T1 ~0.3s, T0-T3 ~2.7s (window responsive during load).
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PUPIL_TRACKING_SILENT"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    import tkinter as tk
    from pupil_tracking.interface.theme import DarkTheme

    print("=" * 66)
    print("GUI STARTUP LATENCY TEST (background detector init)")
    print("=" * 66)

    t0 = time.perf_counter()

    root = tk.Tk()
    root.withdraw()
    t_window = (time.perf_counter() - t0) * 1000.0

    colors = DarkTheme.apply(root)

    from pupil_tracking.interface.gui_app import PupilTrackingGUI
    app = PupilTrackingGUI(root, colors=colors)
    t_gui_init = (time.perf_counter() - t0) * 1000.0

    root.deiconify()
    root.update_idletasks()
    t_window_shown = (time.perf_counter() - t0) * 1000.0

    # Trigger _init_detector (launches background thread for UnifiedDetector)
    app._init_detector()
    t_init_called = (time.perf_counter() - t0) * 1000.0

    # Poll until detector ready (event loop must run for root.after callbacks)
    deadline = time.perf_counter() + 30
    while app._detector is None and time.perf_counter() < deadline:
        root.update_idletasks()
        root.update()
        time.sleep(0.05)

    t_detector_ready = (time.perf_counter() - t0) * 1000.0

    print(f"  {'Tk root created':<42} {t_window:8.1f} ms")
    print(f"  {'PupilTrackingGUI.__init__':<42} {t_gui_init:8.1f} ms")
    print(f"  {'Window shown (deiconify)':<42} {t_window_shown:8.1f} ms")
    print(f"  {'_init_detector() called':<42} {t_init_called:8.1f} ms")
    print(f"  {'Detector ready (background)':<42} {t_detector_ready:8.1f} ms")
    print()
    print(f"  Window-to-visible:       {t_window_shown:8.1f} ms")
    print(f"  Init-to-detector-ready:  {t_detector_ready - t_init_called:8.1f} ms")
    print(f"  Total T0-to-ready:       {t_detector_ready:8.1f} ms")
    print("=" * 66)

    root.destroy()


if __name__ == "__main__":
    main()
