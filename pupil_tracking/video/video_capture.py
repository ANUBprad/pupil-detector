# pupil_tracking/video/video_capture.py
"""Threaded frame reader and async camera capture extracted from
:class:`OptimizedVideoProcessor` during the Phase-5 refactoring.

These are generic threading utilities that only depend on stdlib
and OpenCV.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameReader(threading.Thread):
    """Decode-ahead frame reader (S3)."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        q: queue.Queue,
        stride: int = 1,
        max_frames: Optional[int] = None,
    ):
        super().__init__(daemon=True)
        self._cap = cap
        self._q = q
        self._stride = stride
        self._max = max_frames
        self._stop_event = threading.Event()

    def run(self):
        idx = 0
        produced = 0
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                break
            if idx % self._stride == 0:
                try:
                    self._q.put((idx, frame), timeout=10.0)
                except queue.Full:
                    break
                produced += 1
                if self._max and produced >= self._max:
                    break
            idx += 1
        self._q.put(None)  # sentinel

    def stop(self):
        self._stop_event.set()


class AsyncCapture(threading.Thread):
    """Asynchronous camera capture thread.

    Captures frames from camera in background thread,
    providing non-blocking frame retrieval for real-time processing.
    """

    def __init__(self, camera_id: int = 0, buffer_size: int = 2):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.buffer_size = buffer_size

        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self._lock = threading.Lock()
        self._exception = None
        self._frame_count = 0

    def run(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self.running = True
            consecutive_failures = 0

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        logger.warning(
                            "Camera read failed %d times - stopping capture",
                            consecutive_failures,
                        )
                        break
                    continue
                consecutive_failures = 0

                self._frame_count += 1
                item = (self._frame_count, frame, time.time())

                try:
                    self.frame_queue.put_nowait(item)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(item)
                    except queue.Full:
                        pass

        except Exception as e:
            with self._lock:
                self._exception = e
            logger.error("AsyncCapture error: %s", e)
        finally:
            if self.cap is not None:
                self.cap.release()
            self.running = False

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            item = self.frame_queue.get(timeout=timeout)
            if isinstance(item, tuple):
                return item[1]
            return item
        except queue.Empty:
            return None

    def read(self, timeout: float = 0.1):
        latest = None
        try:
            latest = self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def stop(self) -> None:
        self.running = False
        if self.is_alive():
            self.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    def get_error(self) -> Optional[Exception]:
        with self._lock:
            return self._exception
