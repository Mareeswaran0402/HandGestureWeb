import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";


// ============================================================
// SETTINGS
// ============================================================

const GESTURES = [
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
];

// Minimum prediction confidence
const CONFIDENCE_THRESHOLD = 55;

// Difference between best and second-best prediction
const MARGIN_THRESHOLD = 8;

// Gesture must be held for 0.3 seconds
const HOLD_TIME = 300;

// Number of stable predictions required
const STABLE_FRAMES = 2;

// EMA smoothing factor
const EMA_ALPHA = 0.45;

// Number of recent predictions to keep
const HISTORY_SIZE = 6;

// Prevent rapid switching between gestures
const SWITCH_MARGIN = 10;

// Send landmarks to Flask every 300 ms
const SEND_INTERVAL = 300;


// ============================================================
// HTML ELEMENTS
// ============================================================

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const detectedGesture =
    document.getElementById("detectedGesture");

const confidenceText =
    document.getElementById("confidence");

const confidenceFill =
    document.getElementById("confidenceFill");

const statusText =
    document.getElementById("status");

const sentenceBox =
    document.getElementById("sentence");

const addWordBtn =
    document.getElementById("addWordBtn");

const spaceBtn =
    document.getElementById("spaceBtn");

const speakBtn =
    document.getElementById("speakBtn");

const deleteBtn =
    document.getElementById("deleteBtn");

const clearBtn =
    document.getElementById("clearBtn");

const finishBtn =
    document.getElementById("finishBtn");

const ctx = overlay.getContext("2d");


// ============================================================
// GLOBAL VARIABLES
// ============================================================

let handLandmarker = null;

let stream = null;

let cameraRunning = false;

let animationId = null;

let lastVideoTime = -1;

let lastSendTime = 0;

// IMPORTANT:
// Prevent multiple requests to Render at the same time.
let predictionInFlight = false;


// ============================================================
// SENTENCE STATE
// ============================================================

let sentence = "";


// ============================================================
// GESTURE STATE
// ============================================================

let currentGesture = null;

let candidateGesture = null;

let candidateStartTime = 0;

let stableFrameCount = 0;

// Prevent the same gesture from being inserted repeatedly
// while the hand remains in the same position.
let insertedGesture = null;


// ============================================================
// SMOOTHING STATE
// ============================================================

let lastProbabilities = null;

let probabilityHistory = [];


// ============================================================
// MEDIAPIPE MODEL
// ============================================================

async function loadHandLandmarker() {

    try {

        statusText.textContent =
            "Loading hand detection model...";


        // Load MediaPipe WASM files
        const vision =
            await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
            );


        // Create hand landmarker
        handLandmarker =
            await HandLandmarker.createFromOptions(
                vision,
                {
                    baseOptions: {

                        // IMPORTANT:
                        // Flask static folder
                        modelAssetPath:
                            "/static/models/hand_landmarker.task",

                        // Use browser GPU
                        delegate: "GPU"
                    },

                    runningMode: "VIDEO",

                    numHands: 1,

                    minHandDetectionConfidence: 0.5,

                    minHandPresenceConfidence: 0.5,

                    minTrackingConfidence: 0.5
                }
            );


        statusText.textContent =
            "Model loaded. Click Start Camera.";

        console.log(
            "MediaPipe HandLandmarker loaded successfully."
        );

    } catch (error) {

        console.error(
            "MediaPipe loading error:",
            error
        );

        statusText.textContent =
            "Failed to load hand detection model.";
    }
}


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {

    if (!handLandmarker) {

        statusText.textContent =
            "Please wait for the model to load.";

        return;
    }


    try {

        // Ask browser for webcam
        stream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        },

                        facingMode: "user"
                    },

                    audio: false
                }
            );


        video.srcObject = stream;

        await video.play();


        cameraRunning = true;

        lastVideoTime = -1;

        lastSendTime = 0;

        predictionInFlight = false;


        resetPredictionState();


        statusText.textContent =
            "Camera running. Show a gesture.";


        startBtn.disabled = true;

        stopBtn.disabled = false;


        resizeCanvas();


        // Start prediction loop
        predictLoop();

    } catch (error) {

        console.error(
            "Camera error:",
            error
        );

        statusText.textContent =
            "Camera permission denied or camera unavailable.";
    }
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    cameraRunning = false;


    if (animationId) {

        cancelAnimationFrame(animationId);

        animationId = null;
    }


    if (stream) {

        stream.getTracks().forEach(
            track => track.stop()
        );

        stream = null;
    }


    video.srcObject = null;


    clearCanvas();


    detectedGesture.textContent = "—";

    confidenceText.textContent = "0%";

    confidenceFill.style.width = "0%";


    statusText.textContent =
        "Camera stopped.";


    startBtn.disabled = false;

    stopBtn.disabled = true;


    predictionInFlight = false;


    resetPredictionState();
}


// ============================================================
// RESIZE OVERLAY
// ============================================================

function resizeCanvas() {

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {
        return;
    }


    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;
}


// ============================================================
// CLEAR OVERLAY
// ============================================================

function clearCanvas() {

    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );
}


// ============================================================
// HAND CONNECTIONS
// ============================================================

const HAND_CONNECTIONS = [

    // Thumb
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    // Index
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    // Middle
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    // Ring
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    // Pinky
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    // Palm
    [0, 17]
];


// ============================================================
// DRAW 21 LANDMARKS
// ============================================================

function drawHand(landmarks) {

    clearCanvas();


    if (!landmarks) {
        return;
    }


    const width = overlay.width;

    const height = overlay.height;


    // --------------------------------------------------------
    // GREEN CONNECTION LINES
    // --------------------------------------------------------

    ctx.strokeStyle = "lime";

    ctx.lineWidth = 3;


    for (
        const [start, end]
        of HAND_CONNECTIONS
    ) {

        const p1 =
            landmarks[start];

        const p2 =
            landmarks[end];


        ctx.beginPath();


        ctx.moveTo(
            p1.x * width,
            p1.y * height
        );


        ctx.lineTo(
            p2.x * width,
            p2.y * height
        );


        ctx.stroke();
    }


    // --------------------------------------------------------
    // GREEN LANDMARK POINTS
    // --------------------------------------------------------

    for (
        const landmark
        of landmarks
    ) {

        const x =
            landmark.x * width;

        const y =
            landmark.y * height;


        ctx.beginPath();


        ctx.arc(
            x,
            y,
            5,
            0,
            Math.PI * 2
        );


        ctx.fillStyle = "lime";

        ctx.fill();


        ctx.strokeStyle = "black";

        ctx.lineWidth = 1;

        ctx.stroke();
    }
}


// ============================================================
// MAIN CAMERA/PREDICTION LOOP
// ============================================================

function predictLoop() {

    if (!cameraRunning) {
        return;
    }


    if (
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTime
    ) {

        lastVideoTime =
            video.currentTime;


        try {

            // Detect hand in browser
            const result =
                handLandmarker.detectForVideo(
                    video,
                    performance.now()
                );


            // ------------------------------------------------
            // HAND FOUND
            // ------------------------------------------------

            if (
                result.landmarks &&
                result.landmarks.length > 0
            ) {

                const landmarks =
                    result.landmarks[0];


                // Draw green landmarks
                drawHand(landmarks);


                const now =
                    performance.now();


                // ------------------------------------------------
                // SEND ONLY IF:
                //
                // 1. 300 ms passed
                // 2. Previous request finished
                // ------------------------------------------------

                if (
                    now - lastSendTime >=
                        SEND_INTERVAL &&
                    !predictionInFlight
                ) {

                    lastSendTime = now;

                    predictionInFlight = true;


                    // IMPORTANT:
                    // Do NOT await here.
                    // This keeps the camera loop smooth.
                    sendLandmarks(
                        landmarks
                    ).finally(() => {

                        predictionInFlight =
                            false;
                    });
                }

            }

            // ------------------------------------------------
            // NO HAND FOUND
            // ------------------------------------------------

            else {

                clearCanvas();


                // Reset gesture state so the same
                // gesture can be detected again.
                resetGestureTracking();

                resetInsertedGesture();


                detectedGesture.textContent =
                    "—";


                confidenceText.textContent =
                    "0%";


                confidenceFill.style.width =
                    "0%";


                statusText.textContent =
                    "No hand detected.";
            }


        } catch (error) {

            console.error(
                "Prediction loop error:",
                error
            );
        }
    }


    // Continue camera loop
    animationId =
        requestAnimationFrame(
            predictLoop
        );
}


// ============================================================
// SEND LANDMARKS TO FLASK
// ============================================================

async function sendLandmarks(landmarks) {

    try {

        const response =
            await fetch(
                "/predict-landmarks",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    cache: "no-store",

                    body: JSON.stringify(
                        {
                            landmarks:
                                landmarks
                        }
                    )
                }
            );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data =
            await response.json();


        if (!data.success) {

            console.error(
                "Server prediction failed:",
                data.error
            );

            return;
        }


        processPrediction(data);

    } catch (error) {

        console.error(
            "Server prediction error:",
            error
        );
    }
}


// ============================================================
// PROCESS MODEL PREDICTION
// ============================================================

function processPrediction(data) {

    const probabilities =
        Array.isArray(
            data.probabilities
        )
            ? data.probabilities.map(Number)
            : null;


    // Verify 10 probabilities
    if (
        !probabilities ||
        probabilities.length !==
            GESTURES.length
    ) {

        console.error(
            "Invalid probability array:",
            probabilities
        );

        return;
    }


    // --------------------------------------------------------
    // EMA SMOOTHING
    // --------------------------------------------------------

    if (!lastProbabilities) {

        lastProbabilities =
            [...probabilities];

    } else {

        for (
            let i = 0;
            i < probabilities.length;
            i++
        ) {

            lastProbabilities[i] =
                (
                    EMA_ALPHA *
                    probabilities[i]
                ) +
                (
                    (1 - EMA_ALPHA) *
                    lastProbabilities[i]
                );
        }
    }


    // --------------------------------------------------------
    // STORE RECENT HISTORY
    // --------------------------------------------------------

    probabilityHistory.push(
        [...lastProbabilities]
    );


    if (
        probabilityHistory.length >
        HISTORY_SIZE
    ) {

        probabilityHistory.shift();
    }


    // --------------------------------------------------------
    // WEIGHTED SMOOTHING
    // --------------------------------------------------------

    const smoothed =
        getWeightedProbabilities();


    // --------------------------------------------------------
    // SORT GESTURES BY PROBABILITY
    // --------------------------------------------------------

    const sortedIndices =
        [...Array(GESTURES.length).keys()]
            .sort(
                (a, b) =>
                    smoothed[b] -
                    smoothed[a]
            );


    const bestIndex =
        sortedIndices[0];


    const secondIndex =
        sortedIndices[1];


    const bestProbability =
        smoothed[bestIndex] * 100;


    const secondProbability =
        smoothed[secondIndex] * 100;


    const margin =
        bestProbability -
        secondProbability;


    const gesture =
        GESTURES[bestIndex];


    // --------------------------------------------------------
    // UPDATE DISPLAY
    // --------------------------------------------------------

    detectedGesture.textContent =
        gesture;


    confidenceText.textContent =
        `${bestProbability.toFixed(1)}%`;


    confidenceFill.style.width =
        `${Math.min(bestProbability, 100)}%`;


    // --------------------------------------------------------
    // CONFIDENCE CHECK
    // --------------------------------------------------------

    if (
        bestProbability <
        CONFIDENCE_THRESHOLD
    ) {

        statusText.textContent =
            "Low confidence.";


        resetGestureTracking();

        return;
    }


    // --------------------------------------------------------
    // MARGIN CHECK
    // --------------------------------------------------------

    if (
        margin <
        MARGIN_THRESHOLD
    ) {

        statusText.textContent =
            "Prediction uncertain.";


        resetGestureTracking();

        return;
    }


    // --------------------------------------------------------
    // STABLE GESTURE
    // --------------------------------------------------------

    updateStableGesture(
        gesture,
        bestProbability
    );
}


// ============================================================
// WEIGHTED PROBABILITY SMOOTHING
// ============================================================

function getWeightedProbabilities() {

    const result =
        new Array(
            GESTURES.length
        ).fill(0);


    if (
        probabilityHistory.length === 0
    ) {

        return result;
    }


    let totalWeight = 0;


    for (
        let i = 0;
        i < probabilityHistory.length;
        i++
    ) {

        // Newer frames have greater weight
        const weight =
            i + 1;


        totalWeight += weight;


        for (
            let j = 0;
            j < GESTURES.length;
            j++
        ) {

            result[j] +=
                probabilityHistory[i][j] *
                weight;
        }
    }


    // Normalize
    for (
        let i = 0;
        i < result.length;
        i++
    ) {

        result[i] /=
            totalWeight;
    }


    return result;
}


// ============================================================
// STABLE GESTURE + HOLD TIME + HYSTERESIS
// ============================================================

function updateStableGesture(
    gesture,
    confidence
) {

    // --------------------------------------------------------
    // SAME CANDIDATE
    // --------------------------------------------------------

    if (
        candidateGesture ===
        gesture
    ) {

        stableFrameCount++;
    }


    // --------------------------------------------------------
    // NEW CANDIDATE
    // --------------------------------------------------------

    else {

        candidateGesture =
            gesture;

        candidateStartTime =
            performance.now();

        stableFrameCount = 1;
    }


    // --------------------------------------------------------
    // HYSTERESIS
    // --------------------------------------------------------

    if (
        currentGesture &&
        currentGesture !== gesture
    ) {

        const currentIndex =
            GESTURES.indexOf(
                currentGesture
            );


        const newIndex =
            GESTURES.indexOf(
                gesture
            );


        let currentProbability = 0;

        let newProbability =
            confidence;


        if (
            lastProbabilities &&
            currentIndex >= 0
        ) {

            currentProbability =
                lastProbabilities[
                    currentIndex
                ] * 100;
        }


        if (
            lastProbabilities &&
            newIndex >= 0
        ) {

            newProbability =
                lastProbabilities[
                    newIndex
                ] * 100;
        }


        const difference =
            newProbability -
            currentProbability;


        // Don't switch too easily
        if (
            difference <
                SWITCH_MARGIN &&
            currentProbability >=
                CONFIDENCE_THRESHOLD
        ) {

            statusText.textContent =
                `${currentGesture} maintained`;

            return;
        }
    }


    // --------------------------------------------------------
    // HOW LONG HAS THE GESTURE BEEN HELD?
    // --------------------------------------------------------

    const heldTime =
        performance.now() -
        candidateStartTime;


    // --------------------------------------------------------
    // ACCEPT AFTER 0.3 SECONDS
    // --------------------------------------------------------

    if (
        stableFrameCount >=
            STABLE_FRAMES &&
        heldTime >=
            HOLD_TIME
    ) {

        // Don't add the same gesture repeatedly
        if (
            insertedGesture !==
            gesture
        ) {

            addGestureToSentence(
                gesture
            );


            insertedGesture =
                gesture;
        }


        currentGesture =
            gesture;


        statusText.textContent =
            `${gesture} added`;

    } else {

        const remaining =
            Math.max(
                0,
                HOLD_TIME - heldTime
            );


        statusText.textContent =
            `${gesture} — hold ${(remaining / 1000).toFixed(1)}s`;
    }
}


// ============================================================
// RESET GESTURE TRACKING
// ============================================================

function resetGestureTracking() {

    candidateGesture = null;

    candidateStartTime = 0;

    stableFrameCount = 0;

    currentGesture = null;
}


// ============================================================
// RESET INSERTED GESTURE
// ============================================================

function resetInsertedGesture() {

    insertedGesture = null;

    currentGesture = null;
}


// ============================================================
// RESET ALL PREDICTION STATE
// ============================================================

function resetPredictionState() {

    currentGesture = null;

    candidateGesture = null;

    candidateStartTime = 0;

    stableFrameCount = 0;

    insertedGesture = null;

    lastProbabilities = null;

    probabilityHistory = [];
}


// ============================================================
// ADD GESTURE TO SENTENCE
// ============================================================

function addGestureToSentence(
    gesture
) {

    if (!gesture) {
        return;
    }


    // Add a space automatically between words
    if (
        sentence.length > 0 &&
        !sentence.endsWith(" ")
    ) {

        sentence += " ";
    }


    sentence += gesture;


    updateSentenceDisplay();
}


// ============================================================
// UPDATE SENTENCE DISPLAY
// ============================================================

function updateSentenceDisplay() {

    if (
        sentence.length === 0
    ) {

        sentenceBox.textContent =
            "Your sentence will appear here...";

    } else {

        sentenceBox.textContent =
            sentence;
    }
}


// ============================================================
// ADD WORD BUTTON
// ============================================================

addWordBtn.addEventListener(
    "click",
    () => {

        const gesture =
            detectedGesture.textContent
                .trim();


        if (
            gesture &&
            GESTURES.includes(
                gesture
            )
        ) {

            addGestureToSentence(
                gesture
            );


            // Prevent immediate duplicate
            insertedGesture =
                gesture;


            statusText.textContent =
                `${gesture} added manually.`;
        }
    }
);


// ============================================================
// SPACE BUTTON
// ============================================================

spaceBtn.addEventListener(
    "click",
    () => {

        if (
            sentence.length > 0 &&
            !sentence.endsWith(" ")
        ) {

            sentence += " ";

            updateSentenceDisplay();
        }
    }
);


// ============================================================
// SPEAK BUTTON
// ============================================================

speakBtn.addEventListener(
    "click",
    () => {

        speakSentence();
    }
);


// ============================================================
// BROWSER TEXT TO SPEECH
// ============================================================

function speakSentence() {

    const text =
        sentence.trim();


    if (!text) {

        statusText.textContent =
            "Nothing to speak.";

        return;
    }


    if (
        !("speechSynthesis" in window)
    ) {

        statusText.textContent =
            "Speech synthesis is not supported by this browser.";

        return;
    }


    // Stop any existing speech
    window.speechSynthesis.cancel();


    const utterance =
        new SpeechSynthesisUtterance(
            text
        );


    utterance.rate = 0.9;

    utterance.pitch = 1.0;

    utterance.volume = 1.0;


    utterance.onstart = () => {

        statusText.textContent =
            "Speaking...";
    };


    utterance.onend = () => {

        statusText.textContent =
            "Speech finished.";
    };


    utterance.onerror = () => {

        statusText.textContent =
            "Speech error.";
    };


    window.speechSynthesis.speak(
        utterance
    );
}


// ============================================================
// DELETE LAST WORD
// ============================================================

deleteBtn.addEventListener(
    "click",
    () => {

        sentence =
            sentence.trimEnd();


        const lastSpace =
            sentence.lastIndexOf(" ");


        if (
            lastSpace === -1
        ) {

            sentence = "";

        } else {

            sentence =
                sentence.substring(
                    0,
                    lastSpace
                );
        }


        updateSentenceDisplay();


        statusText.textContent =
            "Last word deleted.";
    }
);


// ============================================================
// CLEAR SENTENCE
// ============================================================

clearBtn.addEventListener(
    "click",
    () => {

        sentence = "";


        updateSentenceDisplay();


        statusText.textContent =
            "Sentence cleared.";
    }
);


// ============================================================
// FINISH & SPEAK
// ============================================================

finishBtn.addEventListener(
    "click",
    () => {

        if (
            !sentence.trim()
        ) {

            statusText.textContent =
                "Sentence is empty.";

            return;
        }


        speakSentence();


        statusText.textContent =
            "Sentence finished.";
    }
);


// ============================================================
// CAMERA BUTTONS
// ============================================================

startBtn.addEventListener(
    "click",
    startCamera
);


stopBtn.addEventListener(
    "click",
    stopCamera
);


// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
    "resize",
    resizeCanvas
);


// ============================================================
// INITIAL STATE
// ============================================================

startBtn.disabled = false;

stopBtn.disabled = true;

updateSentenceDisplay();

statusText.textContent =
    "Loading hand detection model...";


// ============================================================
// LOAD MEDIAPIPE
// ============================================================

loadHandLandmarker();