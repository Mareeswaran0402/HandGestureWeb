from __future__ import annotations

import base64
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "gesture_model.pkl"
HAND_MODEL_FILE = BASE_DIR / "hand_landmarker.task"

# The web app intentionally supports only these classes.
GESTURES = [
    "HELLO",
    "STOP",
    "GOOD",
    "YES",
    "PEACE",
    "LOVE",
    "HELP",
    "THANKS",
    "COME",
    "BYE",
]

# Web-side decision threshold requested by the user.
CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.08

app = Flask(__name__)


# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Missing model file: {MODEL_FILE}. "
        "Copy gesture_model.pkl into the web project folder."
    )

if not HAND_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Missing MediaPipe model: {HAND_MODEL_FILE}. "
        "Copy hand_landmarker.task into the web project folder."
    )

model = joblib.load(MODEL_FILE)
model_classes = [str(c) for c in model.classes_]

# Keep only the requested 10 gestures. If an old model still contains
# UNKNOWN, it is ignored rather than exposed by the website.
known_indices = [
    model_classes.index(g)
    for g in GESTURES
    if g in model_classes
]
known_gestures = [
    g for g in GESTURES
    if g in model_classes
]

if not known_gestures:
    raise RuntimeError(
        "The model does not contain any of the expected 10 gestures. "
        "Retrain gesture_model.pkl with the 10-class train.py."
    )

print("Loaded gesture classes:")
for g in known_gestures:
    print(" -", g)

if "UNKNOWN" in model_classes:
    print("Note: UNKNOWN exists in the old model but is ignored by the web app.")


# -----------------------------------------------------------------------------
# MEDIAPIPE
# -----------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_FILE)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")

        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image received.",
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Could not decode the camera frame.",
            }), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return jsonify({
                "success": True,
                "gesture": "NO HAND",
                "confidence": 0.0,
                "margin": 0.0,
                "landmarks": [],
            })

        hand = result.hand_landmarks[0]

        features = []
        for landmark in hand:
            features.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            ])

        if len(features) != 63:
            return jsonify({
                "success": False,
                "error": "Expected 63 hand-landmark features.",
            }), 500

        features_np = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        probabilities_all = model.predict_proba(features_np)[0]

        # Restrict the web application to the requested ten gestures.
        known_probabilities = np.array(
            [probabilities_all[i] for i in known_indices],
            dtype=np.float32,
        )

        if known_probabilities.size == 0:
            return jsonify({
                "success": False,
                "error": "No supported gestures are present in the model.",
            }), 500

        order = np.argsort(known_probabilities)[::-1]
        best_position = int(order[0])
        second_position = int(order[1]) if len(order) > 1 else best_position

        prediction = known_gestures[best_position]
        confidence = float(known_probabilities[best_position])
        second_confidence = float(known_probabilities[second_position])
        margin = confidence - second_confidence

        # The model keeps the original probability scale. This means an old
        # UNKNOWN class cannot be silently renormalized into a known gesture.
        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            display_prediction = "UNCERTAIN"
        else:
            display_prediction = prediction

        landmarks = [
            {
                "x": float(landmark.x),
                "y": float(landmark.y),
            }
            for landmark in hand
        ]

        return jsonify({
            "success": True,
            "gesture": display_prediction,
            "raw_gesture": prediction,
            "confidence": round(confidence * 100.0, 2),
            "margin": round(margin * 100.0, 2),
            "landmarks": landmarks,
        })

    except (ValueError, TypeError, base64.binascii.Error) as exc:
        return jsonify({
            "success": False,
            "error": f"Invalid request: {exc}",
        }), 400
    except Exception as exc:
        print("Prediction error:", exc)
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# -----------------------------------------------------------------------------
# START
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("        HAND GESTURE TO WORD WEBSITE")
    print("=" * 60)
    print("Supported gestures:")
    for gesture in GESTURES:
        print(" -", gesture)
    print("Confidence threshold: 55%")
    print("Hold time: 0.3 seconds")
    print("Open: http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
