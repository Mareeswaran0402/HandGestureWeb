HAND GESTURE TO WORDS - WEB VERSION
===================================

1. Copy these two files from your existing project into this folder:
   - gesture_model.pkl
   - hand_landmarker.task

2. Install dependencies:
   python -m pip install -r requirements.txt

3. Start the website:
   python app.py

4. Open in Chrome/Edge:
   http://127.0.0.1:5000

Important:
- The website supports only these 10 classes:
  HELLO, STOP, GOOD, YES, PEACE, LOVE, HELP, THANKS, COME, BYE
- Confidence threshold: 55%
- Automatic hold time: 0.3 seconds
- If the old model contains UNKNOWN, the web app ignores UNKNOWN.
- For best results, retrain gesture_model.pkl using the 10-class train.py.
- The video and landmark overlay are mirrored together in CSS, so the green
  skeleton should stay aligned with the hand.
