from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_FILE = "gesture_model.pkl"

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

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.08


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Could not find {MODEL_FILE}. "
        f"Make sure it is in the project folder."
    )


model = joblib.load(MODEL_FILE)


# Get the class names stored inside the trained model
MODEL_CLASSES = [
    str(c)
    for c in model.classes_
]


print()
print("=" * 60)
print("GESTURE MODEL LOADED")
print("=" * 60)

print()
print("Model classes:")

for gesture in MODEL_CLASSES:
    print(" -", gesture)

print()
print("Browser MediaPipe: ENABLED")
print("Server MediaPipe: DISABLED")
print("Confidence threshold: 55%")
print("Margin threshold: 8%")
print("=" * 60)
print()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "browser_landmarks": True,
        "server_mediapipe": False,
        "gestures": GESTURES
    })


# ============================================================
# PREDICT FROM BROWSER LANDMARKS
# ============================================================

@app.route(
    "/predict-landmarks",
    methods=["POST"]
)
def predict_landmarks():

    try:

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400


        landmarks = data.get(
            "landmarks"
        )


        # ----------------------------------------------------
        # Validate landmarks
        # ----------------------------------------------------

        if not landmarks:

            return jsonify({
                "success": False,
                "error": "No landmarks received."
            }), 400


        if len(landmarks) != 21:

            return jsonify({
                "success": False,
                "error":
                    f"Expected 21 landmarks, "
                    f"received {len(landmarks)}."
            }), 400


        # ----------------------------------------------------
        # Convert 21 landmarks to 63 features
        #
        # Each landmark:
        # x
        # y
        # z
        #
        # 21 x 3 = 63 features
        # ----------------------------------------------------

        features = []


        for landmark in landmarks:

            x = float(
                landmark.get(
                    "x",
                    0
                )
            )

            y = float(
                landmark.get(
                    "y",
                    0
                )
            )

            z = float(
                landmark.get(
                    "z",
                    0
                )
            )


            features.extend([
                x,
                y,
                z
            ])


        # ----------------------------------------------------
        # Validate feature count
        # ----------------------------------------------------

        if len(features) != 63:

            return jsonify({
                "success": False,
                "error":
                    f"Expected 63 features, "
                    f"received {len(features)}."
            }), 400


        # ----------------------------------------------------
        # Convert to NumPy array
        # ----------------------------------------------------

        X = np.asarray(
            features,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # Random Forest prediction
        # ----------------------------------------------------

        probabilities = model.predict_proba(X)[0]


        # ----------------------------------------------------
        # Keep ONLY the 10 requested gestures
        # ----------------------------------------------------

        filtered_probabilities = []


        for gesture in GESTURES:

            if gesture in MODEL_CLASSES:

                index = MODEL_CLASSES.index(
                    gesture
                )

                probability = float(
                    probabilities[index]
                )

                filtered_probabilities.append(
                    probability
                )

            else:

                filtered_probabilities.append(
                    0.0
                )


        # ----------------------------------------------------
        # Normalize probabilities
        # ----------------------------------------------------

        filtered_probabilities = np.asarray(
            filtered_probabilities,
            dtype=np.float32
        )


        total = float(
            filtered_probabilities.sum()
        )


        if total > 0:

            filtered_probabilities /= total


        # ----------------------------------------------------
        # Find best gesture
        # ----------------------------------------------------

        sorted_indices = np.argsort(
            filtered_probabilities
        )[::-1]


        best_index = int(
            sorted_indices[0]
        )


        if len(sorted_indices) > 1:

            second_index = int(
                sorted_indices[1]
            )

        else:

            second_index = best_index


        best_probability = float(
            filtered_probabilities[
                best_index
            ]
        )


        second_probability = float(
            filtered_probabilities[
                second_index
            ]
        )


        # ----------------------------------------------------
        # Convert to percentages
        # ----------------------------------------------------

        confidence = (
            best_probability * 100
        )


        second_confidence = (
            second_probability * 100
        )


        margin = (
            confidence -
            second_confidence
        )


        raw_gesture = GESTURES[
            best_index
        ]


        # ----------------------------------------------------
        # Acceptance conditions
        #
        # Confidence >= 55%
        # Margin >= 8%
        # ----------------------------------------------------

        accepted = (
            confidence >=
                CONFIDENCE_THRESHOLD * 100
            and
            margin >=
                MARGIN_THRESHOLD * 100
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "probabilities": [
                round(
                    float(p),
                    6
                )
                for p in filtered_probabilities
            ],

            "raw_gesture":
                raw_gesture,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "margin":
                round(
                    margin,
                    2
                ),

            "accepted":
                accepted
        })


    except Exception as error:

        print(
            "Prediction error:",
            error
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )