import mediapipe as mp

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
BaseOptions = mp.tasks.BaseOptions
VisionTaskRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

CONNECTIONS = {
    "tesselation": [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION],
    "contours": [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS],
    "left_iris": [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS],
    "right_iris": [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS],
}


class FaceLandmarkEngine:
    def __init__(self, model_path: str = MODEL_PATH):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, rgb_frame) -> dict:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return {"faces": []}

        faces = []
        for face_landmarks in results.face_landmarks:
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmarks]
            faces.append({"landmarks": landmarks})

        return {"faces": faces, "connections": CONNECTIONS}

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
