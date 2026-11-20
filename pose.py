"""MediaPipe adapter and confidence-gated landmark filtering."""
import cv2
import mediapipe as mp

from kinematics import OneEuro


def open_landmarker(model):
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


class PoseStream:
    def __init__(self, landmarker, confidence=0.65):
        self.landmarker = landmarker
        self.confidence = confidence
        self.filters = {}

    def read(self, frame, time_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, time_ms)
        height, width = frame.shape[:2]
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
        points = {}
        for index, lm in enumerate(landmarks):
            if min(lm.visibility or 0, lm.presence or 0) < self.confidence:
                continue
            if not (0 <= lm.x < 1 and 0 <= lm.y < 1):
                continue
            # Use a common length unit for x and y, invariant to video resolution.
            if index not in self.filters:
                self.filters[index] = OneEuro(beta=2.0), OneEuro(beta=2.0)
            fx, fy = self.filters[index]
            time = time_ms / 1000
            points[index] = (
                fx.update(lm.x * width / height, time) * height,
                fy.update(lm.y, time) * height,
            )
        # A returning landmark starts fresh after an occlusion.
        self.filters = {index: self.filters[index] for index in points}
        return points
