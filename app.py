from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import joblib
import numpy as np
import base64
import os
import threading


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

# Maximum request size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "gesture_model.pkl"
HAND_MODEL_FILE = "hand_landmarker.task"


# =========================================================
# SUPPORTED GESTURES
# =========================================================

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
    "BYE"
]


# =========================================================
# SETTINGS
# =========================================================

# Minimum confidence for accepting a gesture
CONFIDENCE_THRESHOLD = 55.0

# Minimum difference between top and second prediction
MARGIN_THRESHOLD = 8.0


# =========================================================
# LOAD RANDOM FOREST MODEL
# =========================================================

try:

    model = joblib.load(
        MODEL_FILE
    )

except Exception as e:

    print()
    print("=" * 60)
    print("ERROR: COULD NOT LOAD MODEL")
    print("=" * 60)
    print()
    print("File:", MODEL_FILE)
    print("Error:", e)
    print()

    raise SystemExit(1)


# =========================================================
# DISPLAY MODEL INFORMATION
# =========================================================

print()
print("=" * 60)
print("GESTURE MODEL LOADED")
print("=" * 60)
print()
print("Model classes:")

for class_name in model.classes_:

    print(
        " -",
        class_name
    )

print()


# =========================================================
# CHECK EXPECTED GESTURES
# =========================================================

model_class_names = [
    str(class_name)
    for class_name in model.classes_
]


missing_classes = [

    gesture

    for gesture in GESTURES

    if gesture not in model_class_names

]


if missing_classes:

    print(
        "WARNING: Missing gesture classes:"
    )

    for gesture in missing_classes:

        print(
            " -",
            gesture
        )

    print()


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = (
    mp.tasks.BaseOptions
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# =========================================================
# LANDMARKER OPTIONS
# =========================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=HAND_MODEL_FILE

    ),

    running_mode=(
        VisionRunningMode.IMAGE
    ),

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5

)


# =========================================================
# CREATE LANDMARKER
# =========================================================

try:

    landmarker = (
        HandLandmarker.create_from_options(
            options
        )
    )

except Exception as e:

    print()
    print("=" * 60)
    print("ERROR: COULD NOT LOAD MEDIAPIPE MODEL")
    print("=" * 60)
    print()
    print(
        "Make sure this file exists:"
    )
    print(
        HAND_MODEL_FILE
    )
    print()
    print(
        "Error:",
        e
    )
    print()

    raise SystemExit(1)


# =========================================================
# THREAD LOCK
# =========================================================

# Prevent multiple requests from using the same
# MediaPipe landmarker simultaneously.

landmarker_lock = threading.Lock()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": True,

        "mediapipe_loaded": True,

        "gestures": GESTURES

    })


# =========================================================
# PREDICT API
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # =================================================
        # GET JSON
        # =================================================

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "error": "No JSON data received"

            }), 400


        if "image" not in data:

            return jsonify({

                "success": False,

                "error": "No image received"

            }), 400


        image_data = data["image"]


        if not image_data:

            return jsonify({

                "success": False,

                "error": "Image is empty"

            }), 400


        # =================================================
        # REMOVE DATA URL PREFIX
        # =================================================

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]


        # =================================================
        # DECODE BASE64
        # =================================================

        try:

            image_bytes = base64.b64decode(
                image_data,
                validate=True
            )

        except Exception:

            return jsonify({

                "success": False,

                "error": "Invalid Base64 image"

            }), 400


        # =================================================
        # CONVERT TO NUMPY
        # =================================================

        image_array = np.frombuffer(

            image_bytes,

            dtype=np.uint8

        )


        # =================================================
        # DECODE IMAGE
        # =================================================

        frame = cv2.imdecode(

            image_array,

            cv2.IMREAD_COLOR

        )


        if frame is None:

            return jsonify({

                "success": False,

                "error": "Could not decode image"

            }), 400


        # =================================================
        # BGR -> RGB
        # =================================================

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        # =================================================
        # CREATE MEDIAPIPE IMAGE
        # =================================================

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame

        )


        # =================================================
        # MEDIAPIPE HAND DETECTION
        # =================================================

        with landmarker_lock:

            result = landmarker.detect(
                mp_image
            )


        # =================================================
        # NO HAND
        # =================================================

        if not result.hand_landmarks:

            return jsonify({

                "success": True,

                "gesture": "NO HAND",

                "confidence": 0.0,

                "margin": 0.0,

                "landmarks": []

            })


        # =================================================
        # GET FIRST HAND
        # =================================================

        hand = result.hand_landmarks[0]


        # =================================================
        # EXTRACT 63 FEATURES
        # =================================================

        features = []


        for landmark in hand:

            features.append(
                float(landmark.x)
            )

            features.append(
                float(landmark.y)
            )

            features.append(
                float(landmark.z)
            )


        # =================================================
        # VALIDATE FEATURE COUNT
        # =================================================

        if len(features) != 63:

            return jsonify({

                "success": False,

                "error": (
                    "Expected 63 hand features, "
                    f"got {len(features)}"
                )

            }), 500


        # =================================================
        # NUMPY FEATURE ARRAY
        # =================================================

        features = np.array(

            features,

            dtype=np.float32

        ).reshape(

            1,
            -1

        )


        # =================================================
        # MODEL PROBABILITIES
        # =================================================

        probabilities = model.predict_proba(

            features

        )[0]


        # =================================================
        # REMOVE UNKNOWN FROM CONSIDERATION
        # =================================================

        # Even if an old model file still contains
        # UNKNOWN, it will never be displayed.

        valid_indices = [

            index

            for index, class_name
            in enumerate(model_class_names)

            if class_name in GESTURES

        ]


        if not valid_indices:

            return jsonify({

                "success": False,

                "error": (
                    "Model does not contain any "
                    "of the 10 supported gestures."
                )

            }), 500


        # =================================================
        # FILTER TO 10 SUPPORTED CLASSES
        # =================================================

        valid_probabilities = np.array(

            [
                probabilities[index]
                for index in valid_indices
            ],

            dtype=np.float32

        )


        valid_class_names = [

            model_class_names[index]

            for index in valid_indices

        ]


        # =================================================
        # RENORMALIZE
        # =================================================

        probability_sum = (
            valid_probabilities.sum()
        )


        if probability_sum > 0:

            valid_probabilities /= (
                probability_sum
            )


        # =================================================
        # SORT PREDICTIONS
        # =================================================

        sorted_indices = np.argsort(

            valid_probabilities

        )[::-1]


        # =================================================
        # TOP PREDICTION
        # =================================================

        best_position = int(

            sorted_indices[0]

        )


        prediction = (

            valid_class_names[
                best_position
            ]

        )


        best_probability = float(

            valid_probabilities[
                best_position
            ]

        )


        # =================================================
        # SECOND PREDICTION
        # =================================================

        if len(sorted_indices) > 1:

            second_position = int(

                sorted_indices[1]

            )


            second_probability = float(

                valid_probabilities[
                    second_position
                ]

            )

        else:

            second_probability = 0.0


        # =================================================
        # PERCENTAGES
        # =================================================

        confidence = (

            best_probability * 100.0

        )


        second_confidence = (

            second_probability * 100.0

        )


        # =================================================
        # MARGIN
        # =================================================

        margin = (

            confidence
            - second_confidence

        )


        # =================================================
        # CONFIDENCE CHECK
        # =================================================

        accepted = True


        if confidence < CONFIDENCE_THRESHOLD:

            accepted = False


        # =================================================
        # MARGIN CHECK
        # =================================================

        if margin < MARGIN_THRESHOLD:

            accepted = False


        # =================================================
        # UNCERTAIN
        # =================================================

        if not accepted:

            prediction = "UNCERTAIN"


        # =================================================
        # LANDMARK DATA FOR FRONTEND
        # =================================================

        landmarks = []


        for landmark in hand:

            landmarks.append({

                "x": float(
                    landmark.x
                ),

                "y": float(
                    landmark.y
                ),

                "z": float(
                    landmark.z
                )

            })


        # =================================================
        # RETURN RESULT
        # =================================================

        return jsonify({

            "success": True,

            "gesture": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "margin": round(
                margin,
                2
            ),

            "accepted": accepted,

            "landmarks": landmarks

        })


    # =====================================================
    # BASE64 / REQUEST ERROR
    # =====================================================

    except Exception as e:

        print()
        print(
            "Prediction error:",
            e
        )
        print()


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =========================================================
# 413 ERROR
# =========================================================

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({

        "success": False,

        "error": "Image request is too large"

    }), 413


# =========================================================
# SERVER SHUTDOWN
# =========================================================

@app.route(
    "/shutdown",
    methods=["POST"]
)
def shutdown():

    return jsonify({

        "success": False,

        "error": "Shutdown disabled"

    }), 403


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # PORT
    # -----------------------------------------------------

    port = int(

        os.environ.get(
            "PORT",
            5000
        )

    )


    print()
    print("=" * 60)
    print("          HAND GESTURE TO WORDS")
    print("=" * 60)
    print()

    print(
        "Supported gestures:"
    )

    for gesture in GESTURES:

        print(
            " -",
            gesture
        )

    print()

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}%"
    )

    print(
        f"Margin threshold: "
        f"{MARGIN_THRESHOLD}%"
    )

    print(
        f"Port: {port}"
    )

    print()

    print(
        "Starting Flask server..."
    )

    print()


    # -----------------------------------------------------
    # IMPORTANT FOR PUBLIC DEPLOYMENT
    # -----------------------------------------------------

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )