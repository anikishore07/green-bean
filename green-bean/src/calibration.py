"""
Camera calibration and pose estimation (PnP) for basketball tracking.
"""

from typing import Any


class CameraSystem:
    """Handles lens calibration and 2D–3D pose solving (PnP)."""

    def calibrate_lens(self, checkerboard_images: list[Any]) -> tuple[Any, Any]:
        """
        Calibrate the camera lens using checkerboard images.

        Args:
            checkerboard_images: List of images (e.g. numpy arrays in BGR) each
                containing a visible checkerboard. Used to compute intrinsic
                parameters (camera matrix and distortion coefficients).

        Returns:
            Tuple of (camera_matrix, distortion_coefficients).
            - camera_matrix: 3x3 intrinsic matrix (focal lengths, principal point).
            - distortion_coefficients: Distortion coefficients (e.g. k1, k2, p1, p2, k3).
        """
        pass

    def solve_pnp(
        self,
        image_points_2d: Any,
        world_points_3d: Any,
    ) -> tuple[Any, Any]:
        """
        Solve the Perspective-n-Point (PnP) problem: estimate camera pose from
        2D image points and corresponding 3D world points.

        Args:
            image_points_2d: 2D points in the image (pixel coordinates).
                Shape (N, 2) or Nx2 array (e.g. numpy array or list of (x, y)).
            world_points_3d: Corresponding 3D points in world coordinates (meters).
                Shape (N, 3) or Nx3 array.

        Returns:
            Tuple of (rotation_vector, translation_vector).
            - rotation_vector: 3x1 rotation in axis-angle form (e.g. OpenCV format).
            - translation_vector: 3x1 translation in world units (e.g. meters).
            Together they describe the camera pose (or object pose, depending on convention).
        """
        pass
