import cv2
import mediapipe as mp
import numpy as np

def init_tracker():
    mp_hands = mp.solutions.hands
    # Phase 1, Task 3: track exactly 2 hands with a min detection confidence of 0.7
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing, mp_hands

def extract_landmarks(results) -> np.ndarray:
    """
    Phase 1, Task 5: 
    Extracts the (x, y, z) coordinates of all 21 landmarks for both hands (42 landmarks total).
    Flattens these coordinates into a single 1D array. 
    If only one hand is detected, pads the rest of the array with zeros to maintain a consistent shape.
    """
    # 2 hands * 21 landmarks * 3 coordinates = 126
    flattened_landmarks = np.zeros(2 * 21 * 3)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx > 1: # We only care about up to 2 hands
                break
                
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y, landmark.z])
            
            # Put data into the flattened_landmarks array
            start_idx = idx * 21 * 3
            end_idx = start_idx + (21 * 3)
            flattened_landmarks[start_idx:end_idx] = hand_data
            
    return flattened_landmarks

def main():
    # Phase 1, Task 2: initializes the default webcam using OpenCV
    cap = cv2.VideoCapture(0)
    hands, mp_drawing, mp_hands = init_tracker()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Troubleshooting: Check if your webcam is connected or if permissions are granted for terminal/IDE to use the camera.")
        return

    print("Starting webcam. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Ignoring empty camera frame.")
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, mark the image as not writeable
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)

        # Draw the hand annotations on the image.
        rgb_frame.flags.writeable = True
        
        # Phase 1, Task 4: Draw the hand landmarks and connections on the live feed
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                    
        # Extract flattened landmarks
        landmarks_array = extract_landmarks(results)
        
        # Overlay number of hands detected
        if results.multi_hand_landmarks:
            num_hands = min(len(results.multi_hand_landmarks), 2)
            cv2.putText(frame, f"Hands detected: {num_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('JJK Domain Expansion Tracker', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
