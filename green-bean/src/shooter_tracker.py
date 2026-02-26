"""
Shooter selection and tracking using YOLO pose tracking with persistent IDs.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class ShooterSelector:
    """
    Interactive shooter selection tool that allows clicking on a person
    in the first frame to get their persistent track_id for the video.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        """
        Initialize the shooter selector with a YOLO pose model.

        Args:
            model_path: Path to the YOLO pose model weights file.
        """
        self.model = YOLO(model_path)
        self.click_pos: Optional[Tuple[int, int]] = None

    def _on_click(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        """
        Mouse callback to capture click coordinates.

        Args:
            event: OpenCV mouse event type.
            x: X coordinate of the mouse click.
            y: Y coordinate of the mouse click.
            flags: OpenCV event flags (unused).
            param: User data passed to callback (unused).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_pos = (x, y)

    def select_shooter(self, video_source: str) -> Optional[int]:
        """
        Display the first frame with detected people and let the user
        click on one to select the shooter.

        Args:
            video_source: Path to video file or camera index.

        Returns:
            The track_id of the selected shooter, or None if:
            - No people detected in the first frame
            - User clicked outside all bounding boxes
            - User closed the window without clicking (pressed 'q')
        """
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return None

        # Read first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("Error: Could not read first frame from video")
            return None

        # Run tracking on first frame to detect people with track IDs
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],  # 0 = person class in COCO
            verbose=False
        )[0]

        # Check if any people were detected
        if results.boxes is None or len(results.boxes) == 0:
            print("No people detected in the first frame")
            return None

        # Check if tracking IDs are available
        if results.boxes.id is None:
            print("No tracking IDs available")
            return None

        # Prepare to store bounding boxes with their track IDs
        boxes_with_ids = []
        annotated_frame = frame.copy()

        # Draw bounding boxes and labels for each detected person
        for i in range(len(results.boxes)):
            box = results.boxes.xyxy[i].cpu().numpy()
            track_id = int(results.boxes.id[i].item())
            x1, y1, x2, y2 = map(int, box)

            # Store box info for click matching
            boxes_with_ids.append({
                'box': (x1, y1, x2, y2),
                'track_id': track_id
            })

            # Draw green rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID label above the box
            label = f"ID: {track_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Display the frame and set up mouse callback
        window_name = "Select Shooter - Click on a person (press 'q' to cancel)"
        cv2.imshow(window_name, annotated_frame)
        cv2.setMouseCallback(window_name, self._on_click)

        print(f"Detected {len(boxes_with_ids)} person(s). Click on one to select the shooter.")
        print("Press 'q' to cancel selection.")

        # Reset click position
        self.click_pos = None

        # Wait for user to click or press 'q'
        while self.click_pos is None:
            key = cv2.waitKey(100)
            if key & 0xFF == ord('q'):
                print("Selection cancelled by user")
                cv2.destroyWindow(window_name)
                return None

        # User clicked - match click to bounding box
        click_x, click_y = self.click_pos
        cv2.destroyWindow(window_name)

        # Find which box was clicked
        for box_info in boxes_with_ids:
            x1, y1, x2, y2 = box_info['box']
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                selected_id = box_info['track_id']
                print(f"Selected shooter with track ID: {selected_id}")
                return selected_id

        # Click was outside all boxes
        print("Click was outside all bounding boxes. No shooter selected.")
        return None

    def select_shooter_from_frame(self, frame: np.ndarray) -> Optional[int]:
        """
        Display a single frame with detected people and let the user
        click on one to select (or re-select) the shooter.

        Args:
            frame: A single frame (numpy array) already in memory.

        Returns:
            The track_id of the selected shooter, or None if:
            - No people detected in the frame
            - User clicked outside all bounding boxes
            - User closed the window without clicking (pressed 'q')
        """
        # Run tracking on the provided frame to detect people with track IDs
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],  # 0 = person class in COCO
            verbose=False
        )[0]

        # Check if any people were detected
        if results.boxes is None or len(results.boxes) == 0:
            print("No people detected in the frame")
            return None

        # Check if tracking IDs are available
        if results.boxes.id is None:
            print("No tracking IDs available")
            return None

        # Prepare to store bounding boxes with their track IDs
        boxes_with_ids = []
        annotated_frame = frame.copy()

        # Draw bounding boxes and labels for each detected person
        for i in range(len(results.boxes)):
            box = results.boxes.xyxy[i].cpu().numpy()
            track_id = int(results.boxes.id[i].item())
            x1, y1, x2, y2 = map(int, box)

            # Store box info for click matching
            boxes_with_ids.append({
                'box': (x1, y1, x2, y2),
                'track_id': track_id
            })

            # Draw green rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID label above the box
            label = f"ID: {track_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Display the frame and set up mouse callback
        window_name = "Re-select Shooter - Click on person (press 'q' to cancel)"
        cv2.imshow(window_name, annotated_frame)
        cv2.setMouseCallback(window_name, self._on_click)

        print(f"Detected {len(boxes_with_ids)} person(s). Click on one to re-select the shooter.")
        print("Press 'q' to cancel re-selection.")

        # Reset click position
        self.click_pos = None

        # Wait for user to click or press 'q'
        while self.click_pos is None:
            key = cv2.waitKey(100)
            if key & 0xFF == ord('q'):
                print("Re-selection cancelled by user")
                cv2.destroyWindow(window_name)
                return None

        # User clicked - match click to bounding box
        click_x, click_y = self.click_pos
        cv2.destroyWindow(window_name)

        # Find which box was clicked
        for box_info in boxes_with_ids:
            x1, y1, x2, y2 = box_info['box']
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                selected_id = box_info['track_id']
                print(f"Re-selected shooter with track ID: {selected_id}")
                return selected_id

        # Click was outside all boxes
        print("Click was outside all bounding boxes. No shooter re-selected.")
        return None


def get_shooter_pose(results, shooter_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract pose keypoints for a specific shooter from YOLO tracking results.

    Args:
        results: YOLO tracking results object (from model.track()).
        shooter_id: The track_id of the shooter to extract pose for.

    Returns:
        Tuple of (keypoints, confidences) where:
        - keypoints: numpy array of shape (17, 2) with x, y coordinates
        - confidences: numpy array of shape (17,) with confidence values
        Returns None if shooter_id not found or no pose data available.
    """
    # Check if boxes exist
    if results.boxes is None or len(results.boxes) == 0:
        return None

    # Check if tracking IDs exist
    if results.boxes.id is None:
        return None

    # Check if keypoints exist
    if results.keypoints is None or len(results.keypoints) == 0:
        return None

    # Find the index of the shooter in the results
    shooter_index = None
    for i in range(len(results.boxes.id)):
        if int(results.boxes.id[i].item()) == shooter_id:
            shooter_index = i
            break

    # Shooter not found in this frame
    if shooter_index is None:
        return None

    # Extract keypoints for the shooter
    # keypoints.data shape: (num_detections, num_keypoints, 3)
    # where the last dimension is (x, y, confidence)
    kpt_data = results.keypoints.data[shooter_index].cpu().numpy()

    # Split into coordinates and confidences
    keypoints = kpt_data[:, :2]  # Shape: (17, 2) - x, y coordinates
    confidences = kpt_data[:, 2]  # Shape: (17,) - confidence values

    return (keypoints, confidences)
