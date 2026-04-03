import cv2
import csv
import os
import time
import numpy as np

# Reusing camera and MediaPipe logic from Phase 1
from main import init_tracker, extract_landmarks

CSV_FILE = 'gesture_data.csv'

# Key mappings for labels
KEY_MAP = {
    ord('1'): "infinite_void",
    ord('2'): "malevolent_shrine",
    ord('3'): "hollow_purple",
    ord('4'): "mahoraga",
    ord('0'): "neutral"
}

def load_existing_counts():
    """Reads existing CSV to initialize the counter for the overlay."""
    counts = {
        "infinite_void": 0,
        "malevolent_shrine": 0,
        "hollow_purple": 0,
        "mahoraga": 0,
        "neutral": 0
    }
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            for row in reader:
                if row:
                    label = row[0]
                    if label in counts:
                        counts[label] += 1
    return counts

def save_data(label, landmarks_array):
    """Saves the flattened landmark array and label to CSV."""
    file_exists = os.path.exists(CSV_FILE)
    
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file does not exist
        if not file_exists:
            header = ['label'] + [f'lm_{i}' for i in range(len(landmarks_array))]
            writer.writerow(header)
        
        # Write row: label followed by the flattened coordinates
        row = [label] + landmarks_array.tolist()
        writer.writerow(row)

def main():
    cap = cv2.VideoCapture(0)
    hands, mp_drawing, mp_hands = init_tracker()
    counts = load_existing_counts()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Data Collector. Press 'q' to quit.")
    print("Press '1', '2', '3', '4', or '0' to initiate capture sequence.")

    # State variables for burst recording
    recording_mode = False
    countdown_mode = False
    countdown_start = 0
    recording_label = None
    frames_recorded = 0
    TOTAL_FRAMES = 300

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Ignoring empty camera frame.")
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Extract landmarks using Phase 1 logic
        landmarks_array = extract_landmarks(results)
        
        # Draw the hand landmarks on the live feed
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # Draw live counters on screen
        y_offset = 30
        for i, (label, count) in enumerate(counts.items()):
            text = f"{label}: {count}"
            cv2.putText(frame, text, (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Process Countdown & Recording Logic
        if countdown_mode:
            elapsed = time.time() - countdown_start
            if elapsed < 3.0:
                seconds_left = 3 - int(elapsed)
                # Display massive countdown text
                cv2.putText(frame, str(seconds_left), (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
                cv2.putText(frame, f"GET READY FOR {recording_label.upper()}", 
                            (frame.shape[1]//2 - 250, frame.shape[0]//2 + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                countdown_mode = False
                recording_mode = True
                frames_recorded = 0
                
        elif recording_mode:
            save_data(recording_label, landmarks_array)
            frames_recorded += 1
            counts[recording_label] += 1
            
            # Draw recording indicator
            text = f"RECORDING {recording_label.upper().replace('_', ' ')}... {frames_recorded}/{TOTAL_FRAMES}"
            cv2.putText(frame, text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            
            if frames_recorded >= TOTAL_FRAMES:
                recording_mode = False
                print(f"Finished recording set for {recording_label}. Total samples: {counts[recording_label]}")

        # Show general prompt only when not interacting
        if not countdown_mode and not recording_mode:
            cv2.putText(frame, "Press 0-4 to record data, Q to quit", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Data Collector', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key in KEY_MAP and not countdown_mode and not recording_mode:
            recording_label = KEY_MAP[key]
            countdown_mode = True
            countdown_start = time.time()
            print(f"Initiated countdown for {recording_label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
