"""
Vision system for basketball tracking: ball and pose detection via YOLO.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# COCO class ID for "sports ball"
COCO_SPORTS_BALL_CLASS_ID = 32


class GreenBeanVision:
    """Ball and pose detection using YOLO (detection + pose models)."""

    def __init__(self):
        self.ball_model = YOLO("yolov8n.pt")
        self.pose_model = YOLO("yolov8n-pose.pt")

    def detect_ball(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Run ball tracking on a frame (track enables frame-to-frame memory).

        Args:
            frame: BGR image (numpy array).

        Returns:
            Tuple (box, track_id). box is [x1, y1, x2, y2] as numpy array, or None.
            track_id is the tracker-assigned ID (e.g. 1) or None. (None, None) if no ball found.
        """
        results = self.ball_model.track(frame, persist=True, conf=0.2, verbose=False)[0]
        boxes = results.boxes
        if boxes is None:
            return (None, None)

        best_box = None
        best_id = None
        best_conf = 0.0
        for i, cls in enumerate(boxes.cls):
            if int(cls) == COCO_SPORTS_BALL_CLASS_ID:
                conf = float(boxes.conf[i])
                if conf > best_conf:
                    best_conf = conf
                    best_box = boxes.xyxy[i].cpu().numpy()
                    if boxes.id is not None and i < len(boxes.id):
                        best_id = int(boxes.id[i].item())
                    else:
                        best_id = None

        if best_box is None:
            return (None, None)
        return (best_box, best_id)

    def detect_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run pose detection on a frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            Keypoints for the first detected person as numpy array of shape (N, 3)
            (x, y, confidence). None if no person detected.
        """
        results = self.pose_model(frame, conf=0.5, verbose=False)[0]
        keypoints = results.keypoints
        if keypoints is None or len(keypoints) == 0:
            return None

        # First person: shape (num_keypoints, 3) -> (x, y, conf)
        kpt = keypoints.data[0].cpu().numpy()
        return kpt

    def close(self) -> None:
        """No-op for API compatibility (YOLO models do not require explicit release)."""
        pass
