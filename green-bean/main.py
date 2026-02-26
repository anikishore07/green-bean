"""
Main loop: webcam capture, ball and pose detection, display.
Press 'q' to quit.
"""

from collections import deque

import cv2
import numpy as np

from src.vision import GreenBeanVision
from src.shooter_tracker import ShooterSelector, get_shooter_pose


# Use default webcam (0); change to 1, 2, ... for other cameras
VIDEO_SOURCE = "../assets/Testdata1.mp4"

# COCO pose keypoint indices: 0 nose, 1-4 eyes/ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
# 11-12 hips, 13-14 knees, 15-16 ankles.
# Standard body connections (pairs of keypoint indices).
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # face
    (5, 6),                            # shoulders
    (5, 7), (7, 9),                    # left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 10),                   # right arm
    (5, 11), (6, 12),                  # torso: shoulder -> hip
    (11, 12),                          # hips
    (11, 13), (13, 15),                # left leg: hip -> knee -> ankle
    (12, 14), (14, 16),                # right leg
]

CONF_THRESHOLD = 0.5
COLOR_BALL = (0, 255, 0)      # green (BGR)
COLOR_PATH = (0, 255, 255)   # yellow (BGR)
COLOR_JOINT = (255, 0, 0)     # blue (BGR)
COLOR_SKELETON = (255, 0, 0)  # blue (BGR)
TRACKING_LOST_THRESHOLD = 30  # frames (~1 second at 30fps)


def main():
    vision = GreenBeanVision()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Could not open video source: {VIDEO_SOURCE}")
        return

    ball_path = deque(maxlen=10)

    # Shooter selection: let user click on a person in first frame
    selector = ShooterSelector()
    shooter_id = selector.select_shooter(VIDEO_SOURCE)
    if shooter_id is None:
        print("No shooter selected. Exiting.")
        cap.release()
        return

    tracking_lost_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ball: green rectangle + "Ball" text (and track ID if available)
        ball_bbox, track_id = vision.detect_ball(frame)
        if ball_bbox is not None:
            x1, y1, x2, y2 = map(int, ball_bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ball_path.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BALL, 2)
            label = f"Ball {track_id}" if track_id is not None else "Ball"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BALL, 2)

        # Draw ball path (yellow line through recent centers)
        if len(ball_path) >= 2:
            pts = np.array(ball_path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=False, color=COLOR_PATH, thickness=2)

        # Pose: track all people, filter for selected shooter, draw skeleton
        pose_results = selector.model.track(
            frame, persist=True, tracker="botsort.yaml", classes=[0], verbose=False
        )[0]
        shooter_pose = get_shooter_pose(pose_results, shooter_id)
        if shooter_pose is not None:
            tracking_lost_frames = 0
            keypoints, confidences = shooter_pose
            # Draw joints (blue circles) where confidence > 0.5
            for i in range(len(keypoints)):
                if confidences[i] > CONF_THRESHOLD:
                    pt = (int(keypoints[i][0]), int(keypoints[i][1]))
                    cv2.circle(frame, pt, 5, COLOR_JOINT, -1)
            # Draw skeleton lines between valid joints
            for i, j in SKELETON_CONNECTIONS:
                if i >= len(keypoints) or j >= len(keypoints):
                    continue
                if confidences[i] > CONF_THRESHOLD and confidences[j] > CONF_THRESHOLD:
                    cv2.line(frame,
                             (int(keypoints[i][0]), int(keypoints[i][1])),
                             (int(keypoints[j][0]), int(keypoints[j][1])),
                             COLOR_SKELETON, 2)
        else:
            tracking_lost_frames += 1
            cv2.putText(frame, "Tracking Lost", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if tracking_lost_frames >= TRACKING_LOST_THRESHOLD:
                print("Tracking lost for 1+ seconds. Re-select shooter...")
                new_shooter_id = selector.select_shooter_from_frame(frame)
                if new_shooter_id is not None:
                    shooter_id = new_shooter_id
                    tracking_lost_frames = 0
                    print(f"Re-locked to shooter ID: {shooter_id}")
                else:
                    print("Re-selection failed. Continuing...")

        cv2.imshow("Green Bean - Ball & Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    vision.close()


if __name__ == "__main__":
    main()
