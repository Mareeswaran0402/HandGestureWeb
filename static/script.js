const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const gestureElement = document.getElementById("gesture");
const confidenceElement = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const holdElement = document.getElementById("holdMetric");
const holdBar = document.getElementById("holdBar");
const holdDisplay = document.getElementById("holdDisplay");
const sentenceElement = document.getElementById("sentence");
const wordCountElement = document.getElementById("wordCount");
const statusElement = document.getElementById("status");
const topStatus = document.getElementById("topStatus");
const cameraStatus = document.getElementById("cameraStatus");
const cameraMessage = document.getElementById("cameraMessage");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");


// =========================================================
// SETTINGS
// =========================================================

const HOLD_TIME = 300;              // 0.3 seconds
const SMOOTHING_SIZE = 5;           // recent predictions
const POLL_INTERVAL = 110;          // ms between backend requests


// =========================================================
// STATE
// =========================================================

let stream = null;
let running = false;
let predictionBusy = false;
let lastPredictionTime = 0;

let sentence = [];

let currentGesture = "NO HAND";
let currentConfidence = 0;

let previousGesture = "";
let gestureStartTime = 0;
let gestureAlreadyAdded = false;

let predictionHistory = [];


// =========================================================
// START CAMERA
// =========================================================

async function startCamera() {

    if (running) {
        return;
    }

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;

        await video.play();

        running = true;

        cameraMessage.style.display = "none";

        cameraStatus.textContent = "Camera running";
        statusElement.textContent = "Show your hand";
        topStatus.textContent = "LIVE";

        requestAnimationFrame(cameraLoop);

    } catch (error) {

        console.error("Camera error:", error);

        cameraMessage.style.display = "flex";

        cameraStatus.textContent = "Camera access failed";
        statusElement.textContent = "Allow camera access and try again";
        topStatus.textContent = "CAMERA OFF";

    }
}


// =========================================================
// STOP CAMERA
// =========================================================

function stopCamera() {

    running = false;

    predictionBusy = false;

    if (stream) {

        stream.getTracks().forEach(track => track.stop());

        stream = null;
    }

    video.srcObject = null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    resetRecognitionState();

    cameraMessage.style.display = "flex";

    cameraStatus.textContent = "Camera stopped";
    statusElement.textContent = "Camera stopped";
    topStatus.textContent = "READY";

}


// =========================================================
// RESET RECOGNITION STATE
// =========================================================

function resetRecognitionState() {

    currentGesture = "NO HAND";
    currentConfidence = 0;

    previousGesture = "";
    gestureStartTime = 0;
    gestureAlreadyAdded = false;

    predictionHistory = [];

    updatePredictionUI();
    setHoldProgress(0);
}


// =========================================================
// CAMERA LOOP
// =========================================================

function cameraLoop(timestamp) {

    if (!running) {
        return;
    }

    if (
        video.readyState >= 2 &&
        !predictionBusy &&
        timestamp - lastPredictionTime >= POLL_INTERVAL
    ) {

        lastPredictionTime = timestamp;
        predictionBusy = true;

        predictFrame().finally(() => {
            predictionBusy = false;
        });

    }

    requestAnimationFrame(cameraLoop);
}


// =========================================================
// SEND FRAME TO PYTHON
// =========================================================

async function predictFrame() {

    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) {
        return;
    }

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = width;
    tempCanvas.height = height;

    const tempCtx = tempCanvas.getContext("2d");

    // Send the ORIGINAL frame to Python.
    // The visible video and overlay are mirrored together in CSS.
    tempCtx.drawImage(video, 0, 0, width, height);

    const imageData = tempCanvas.toDataURL("image/jpeg", 0.72);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || "Prediction failed");
        }

        if (result.gesture === "NO HAND") {

            predictionHistory = [];
            resetRecognitionState();
            drawLandmarks([]);
            return;
        }

        // Browser-side smoothing of stable labels.
        predictionHistory.push(result.gesture);

        if (predictionHistory.length > SMOOTHING_SIZE) {
            predictionHistory.shift();
        }

        const stableGesture = getMostCommon(predictionHistory);

        currentGesture = stableGesture;
        currentConfidence = Number(result.confidence || 0);

        updatePredictionUI();
        drawLandmarks(result.landmarks || []);
        handleHold();

    } catch (error) {

        console.error("Prediction error:", error);

        statusElement.textContent = "Server prediction error";
        topStatus.textContent = "ERROR";
    }
}


// =========================================================
// MOST COMMON PREDICTION
// =========================================================

function getMostCommon(items) {

    if (!items.length) {
        return "NO HAND";
    }

    const counts = {};

    for (const item of items) {
        counts[item] = (counts[item] || 0) + 1;
    }

    let best = items[0];
    let bestCount = counts[best];

    for (const item of Object.keys(counts)) {

        if (counts[item] > bestCount) {
            best = item;
            bestCount = counts[item];
        }
    }

    return best;
}


// =========================================================
// DRAW LANDMARKS
// =========================================================

function drawLandmarks(landmarks) {

    if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length !== 21) {
        return;
    }

    const connections = [
        [0,1], [1,2], [2,3], [3,4],
        [0,5], [5,6], [6,7], [7,8],
        [0,9], [9,10], [10,11], [11,12],
        [0,13], [13,14], [14,15], [15,16],
        [0,17], [17,18], [18,19], [19,20],
        [5,9], [9,13], [13,17]
    ];

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const [start, end] of connections) {

        const x1 = landmarks[start].x * canvas.width;
        const y1 = landmarks[start].y * canvas.height;
        const x2 = landmarks[end].x * canvas.width;
        const y2 = landmarks[end].y * canvas.height;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    ctx.fillStyle = "lime";

    for (const point of landmarks) {

        const x = point.x * canvas.width;
        const y = point.y * canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
    }
}


// =========================================================
// HOLD LOGIC
// =========================================================

function handleHold() {

    if (
        currentGesture === "UNCERTAIN" ||
        currentGesture === "NO HAND"
    ) {

        previousGesture = "";
        gestureStartTime = 0;
        gestureAlreadyAdded = false;

        setHoldProgress(0);

        statusElement.textContent =
            currentGesture === "UNCERTAIN"
                ? "Gesture not confident enough"
                : "Show your hand";

        return;
    }

    if (currentGesture !== previousGesture) {

        previousGesture = currentGesture;
        gestureStartTime = Date.now();
        gestureAlreadyAdded = false;
        setHoldProgress(0);

        statusElement.textContent =
            "Hold " + currentGesture;
    }

    const heldTime = Date.now() - gestureStartTime;

    const progress = Math.min(
        (heldTime / HOLD_TIME) * 100,
        100
    );

    setHoldProgress(progress);

    if (
        heldTime >= HOLD_TIME &&
        !gestureAlreadyAdded
    ) {

        sentence.push(currentGesture);

        gestureAlreadyAdded = true;

        updateSentence();

        statusElement.textContent =
            "Added: " + currentGesture;

    }
}


// =========================================================
// HOLD UI
// =========================================================

function setHoldProgress(value) {

    const rounded = Math.max(
        0,
        Math.min(100, value)
    );

    holdElement.textContent =
        `${rounded.toFixed(0)}%`;

    holdDisplay.textContent =
        `Hold: ${rounded.toFixed(0)}%`;

    holdBar.style.width =
        `${rounded}%`;
}


// =========================================================
// PREDICTION UI
// =========================================================

function updatePredictionUI() {

    gestureElement.textContent = currentGesture;

    confidenceElement.textContent =
        `${currentConfidence.toFixed(1)}%`;

    confidenceBar.style.width =
        `${Math.min(currentConfidence, 100)}%`;

    if (currentGesture === "UNCERTAIN") {
        gestureElement.style.color = "#f4ca5d";
        topStatus.textContent = "UNCERTAIN";
    } else if (currentGesture === "NO HAND") {
        gestureElement.style.color = "#ef6670";
        topStatus.textContent = "READY";
    } else {
        gestureElement.style.color = "#35e27f";
        topStatus.textContent = "DETECTING";
    }
}


// =========================================================
// SENTENCE FUNCTIONS
// =========================================================

function getSentenceText() {

    let text = "";

    for (const item of sentence) {

        if (item === "|") {
            text += " ";
            continue;
        }

        if (text.length && !text.endsWith(" ")) {
            text += " ";
        }

        text += item;
    }

    return text.trim();
}


function updateSentence() {

    sentenceElement.textContent =
        getSentenceText();

    wordCountElement.textContent =
        sentence.length;
}


function addCurrentWord() {

    if (
        currentGesture === "NO HAND" ||
        currentGesture === "UNCERTAIN"
    ) {
        statusElement.textContent = "No valid gesture";
        return;
    }

    sentence.push(currentGesture);

    updateSentence();

    gestureAlreadyAdded = true;

    statusElement.textContent =
        "Added: " + currentGesture;
}


function addSpace() {

    if (
        sentence.length &&
        sentence[sentence.length - 1] !== "|"
    ) {
        sentence.push("|");
        updateSentence();
    }
}


function deleteLast() {

    if (!sentence.length) {
        statusElement.textContent = "Sentence is empty";
        return;
    }

    sentence.pop();
    updateSentence();
    statusElement.textContent = "Last item deleted";
}


function clearSentence() {

    sentence = [];
    updateSentence();
    statusElement.textContent = "Sentence cleared";
}


// =========================================================
// SPEECH
// =========================================================

function speakSentence() {

    const text = getSentenceText();

    if (!text) {
        statusElement.textContent = "Nothing to speak";
        return;
    }

    if (!("speechSynthesis" in window)) {
        statusElement.textContent = "Speech is not supported in this browser";
        return;
    }

    window.speechSynthesis.cancel();

    const utterance =
        new SpeechSynthesisUtterance(text);

    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    utterance.onstart = () => {
        statusElement.textContent = "Speaking...";
        topStatus.textContent = "SPEAKING";
    };

    utterance.onend = () => {
        statusElement.textContent = "Ready";
        topStatus.textContent = "LIVE";
    };

    utterance.onerror = (event) => {
        console.error("Speech error:", event);
        statusElement.textContent = "Speech error";
        topStatus.textContent = "ERROR";
    };

    window.speechSynthesis.speak(utterance);
}


function finishSentence() {

    if (!getSentenceText()) {
        statusElement.textContent = "Sentence is empty";
        return;
    }

    speakSentence();
}


// =========================================================
// BUTTONS
// =========================================================

startCameraBtn.addEventListener(
    "click",
    startCamera
);

stopCameraBtn.addEventListener(
    "click",
    stopCamera
);

document.getElementById("addWordBtn").addEventListener(
    "click",
    addCurrentWord
);

document.getElementById("spaceBtn").addEventListener(
    "click",
    addSpace
);

document.getElementById("speakBtn").addEventListener(
    "click",
    speakSentence
);

document.getElementById("deleteBtn").addEventListener(
    "click",
    deleteLast
);

document.getElementById("clearBtn").addEventListener(
    "click",
    clearSentence
);

document.getElementById("finishBtn").addEventListener(
    "click",
    finishSentence
);


// =========================================================
// INITIAL UI
// =========================================================

updateSentence();
setHoldProgress(0);
updatePredictionUI();
