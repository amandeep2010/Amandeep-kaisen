import cv2
import numpy as np
import joblib
from collections import deque
import asyncio
import websockets
import json
from main import init_tracker, extract_landmarks
import threading
import os
import signal
import http

# Global variable to hold the latest stabilized prediction
latest_stable_label = "neutral"

# Track connected clients + shutdown logic
connected_clients = set()
shutdown_task = None

def process_frames():
    global latest_stable_label
    try:
        model = joblib.load('jjk_model.pkl')
    except Exception as e:
        print("Error loading model:", e)
        return

    # In macOS, starting Cap normally grabs the primary camera (often FaceTime / built-in)
    cap = cv2.VideoCapture(0)
    hands, _, mp_hands = init_tracker()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Background MediaPipe CV process running (Headless).")

    prediction_buffer = deque(maxlen=5)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)

        landmarks_array = extract_landmarks(results)
        
        current_prediction = "neutral"
        if results.multi_hand_landmarks:
            pred = model.predict([landmarks_array])[0]
            current_prediction = pred

        prediction_buffer.append(current_prediction)

        stable_label = "neutral"
        if len(prediction_buffer) == 5 and len(set(prediction_buffer)) == 1:
            stable_label = prediction_buffer[0]

        latest_stable_label = stable_label

    cap.release()

async def schedule_shutdown():
    """Wait 5 seconds, then exit if no client has reconnected."""
    print("⏳ No clients connected. Shutting down in 5 seconds (refresh to cancel)...")
    await asyncio.sleep(5)
    if len(connected_clients) == 0:
        print("🔴 No clients reconnected. Shutting down.")
        os.kill(os.getpid(), signal.SIGTERM)

async def handler(websocket):
    global latest_stable_label, shutdown_task
    
    # Register client
    connected_clients.add(websocket)
    print(f"✅ Client connected: {websocket.remote_address} ({len(connected_clients)} active)")
    
    # Cancel any pending shutdown (e.g., page was refreshed)
    if shutdown_task and not shutdown_task.done():
        shutdown_task.cancel()
        print("   ↳ Shutdown cancelled — client reconnected.")
    
    last_sent_label = None
    try:
        while True:
            current_label = latest_stable_label
            if current_label != last_sent_label:
                message = json.dumps({"gesture": current_label})
                await websocket.send(message)
                last_sent_label = current_label
                print(f"Broadcasted -> {message}")
                
            await asyncio.sleep(0.05)  # 20 Hz poll
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Unregister client
        connected_clients.discard(websocket)
        print(f"❌ Client disconnected: {websocket.remote_address} ({len(connected_clients)} active)")
        
        # If no clients left, start shutdown countdown
        if len(connected_clients) == 0:
            shutdown_task = asyncio.ensure_future(schedule_shutdown())

async def process_request(path, request_headers):
    # Reply to non-WebSocket HTTP probes (for example Render health checks).
    upgrade_header = request_headers.get("Upgrade", "")
    if upgrade_header.lower() != "websocket":
        return (http.HTTPStatus.OK, [], b"OK\n")

    return None

async def main_server():
    print("WebSocket API Server listening on 0.0.0.0 (Render PORT or 8765)")
    async with websockets.serve(handler, "0.0.0.0", int(os.environ.get("PORT", 8765))):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    # Start OpenCV webcam extraction in a daemon thread so it doesn't block AsyncIO
    t = threading.Thread(target=process_frames, daemon=True)
    t.start()
    
    try:
        asyncio.run(main_server())
    except (KeyboardInterrupt, SystemExit):
        print("API Server Stopped.")
