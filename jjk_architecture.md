# Architecture Document: JJK Domain Expansion CV Project

## 1. Project Overview
Build a Python desktop application that uses a webcam to track hand landmarks, classifies complex, overlapping hand gestures into specific Jujutsu Kaisen moves using a custom Machine Learning model, and overlays visual text/graphics onto the live video feed.

## 2. Technology Stack
- **Language:** Python 3.10+
- **Computer Vision:** `opencv-python`, `mediapipe` (for multi-hand tracking)
- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`
- **Environment:** `requirements.txt` or `Pipfile` must be generated.

## 3. Project Phases & Agent Directives

### Phase 1: Environment & Multi-Hand Tracking Setup
**Agent Tasks:**
1. Generate `requirements.txt` with the necessary libraries.
2. Create `main.py` that initializes the default webcam using OpenCV.
3. Integrate `mediapipe.solutions.hands` configured to track exactly **2 hands** with a minimum detection confidence of 0.7.
4. Draw the hand landmarks and connections on the live feed.
5. Extract the (x, y, z) coordinates of all 21 landmarks for both hands (42 landmarks total). Flatten these coordinates into a single 1D array. If only one hand is detected, pad the rest of the array with zeros to maintain a consistent shape.

### Phase 2: The Data Collection Tool
**Agent Tasks:**
1. Create a new file called `data_collector.py` (inheriting camera logic from Phase 1).
2. Implement key-press listeners using `cv2.waitKey()` to capture data:
   - Press '1': Save current flattened landmark array to a CSV file `gesture_data.csv` with the label `infinite_void`.
   - Press '2': Save with label `malevolent_shrine`.
   - Press '3': Save with label `hollow_purple`.
   - Press '4': Save with label `mahoraga`.
   - Press '0': Save with label `neutral` (resting hands).
3. Display a live counter on the OpenCV video feed showing how many samples have been collected for each class.

### Phase 3: The Machine Learning Pipeline
**Agent Tasks:**
1. Create `train_model.py`.
2. Load `gesture_data.csv` using Pandas.
3. Split the data into features (X) and labels (y), and then into training and testing sets (80/20 split).
4. Train a `RandomForestClassifier` (from scikit-learn) on the data.
5. Print the accuracy metrics (accuracy score, classification report) to the terminal.
6. Export the trained model to a file named `jjk_model.pkl` using the `pickle` or `joblib` library.

### Phase 4: Live Detection and VFX Integration
**Agent Tasks:**
1. Create `app.py`, integrating the webcam and MediaPipe logic.
2. Load the trained `jjk_model.pkl` classifier.
3. For every frame, extract the flattened hand landmarks and pass them to the model to predict the gesture.
4. Implement a stabilizing logic (e.g., a deque or frame buffer) so a gesture must be predicted consistently for 5 frames before triggering, preventing flickering.
5. **VFX Implementation via OpenCV:**
   - If `infinite_void`: Overlay large, stylized cyan text "DOMAIN EXPANSION: INFINITE VOID" at the top of the frame.
   - If `malevolent_shrine`: Overlay red text "MALEVOLENT SHRINE" and draw a red tint/rectangle border over the frame.
   - If `hollow_purple`: Draw a glowing purple circle (`cv2.circle`) between the coordinates of the two hands.

## 4. Execution Rules for Antigravity Agents
- Run terminal commands autonomously to install dependencies.
- Handle any `cv2` or webcam permission errors gracefully, providing troubleshooting steps.
- Ensure the codebase is highly modular so I can easily swap out VFX or add new hand signs later.