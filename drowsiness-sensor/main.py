# main.py
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Import project modules
import config
from database_manager import init_database, load_all_driver_encodings
from face_recognizer import load_face_models, detect_and_recognize_face
from drowsiness_detector import calculate_ear, dist # Import EAR helpers
from arduino_comm import ArduinoComm
from ui_manager import run_driver_manager_ui

def run_drowsiness_detection(arduino_handler, face_detector, face_embedder, known_encodings):
    """Runs the main drowsiness detection loop with facial recognition and periodic re-verification."""
    print("\nStarting Drowsiness Detection System Loop...")

    # Initialize MediaPipe Face Mesh
    print("Initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = None
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MESH_MAX_FACES,
            refine_landmarks=config.FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONF,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONF
        )
    except Exception as e:
        print(f"Error initializing MediaPipe Face Mesh: {e}")
        return # Cannot proceed without face mesh

    # Initialize Video Capture
    print("Starting video capture...")
    cap = cv2.VideoCapture(0) # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        if face_mesh: face_mesh.close()
        return

    # --- State Variables for Main Loop ---
    frame_count = 0 # Optional: for frame-based logic or debugging
    COUNTER = 0
    TOTAL_ALERTS = 0
    DROWSINESS_ALERT = False
    current_driver_id = None
    current_driver_name = None
    driver_authorized = False
    driver_locked_in = False
    last_verification_time = 0.0

    print("Detection loop running... Press 'q' in the video window to exit.")
    # --- Main Loop ---
    while cap.isOpened():
        current_time = time.time()
        frame_count += 1

        success, frame = cap.read()
        if not success:
            # print("Warning: Failed to read frame.") # Can be noisy
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape
        if image_height == 0 or image_width == 0: continue # Skip invalid frames

        # --- Stage 1: Face Recognition / Verification ---
        perform_recognition = False
        status_text = ""
        status_color = (0, 0, 0)
        status_if_fail = "Unauthorized Driver / No Face"

        if not driver_locked_in:
            perform_recognition = True
        else:
            if current_time - last_verification_time >= config.VERIFICATION_INTERVAL:
                perform_recognition = True
                status_if_fail = "RE-VERIFYING..."
            else:
                perform_recognition = False
                driver_authorized = True # Maintain lock
                status_text = f"Driver (Locked): {current_driver_name}"
                status_color = (0, 255, 0)

        if perform_recognition:
            # Pass loaded models to the recognition function
            detected_id, detected_name, face_box = detect_and_recognize_face(
                frame, face_detector, face_embedder, known_encodings
            )

            recognition_passed = False
            if detected_id is not None:
                if driver_locked_in: # Re-verification check
                    if detected_id == current_driver_id: recognition_passed = True
                    else: print(f"[{time.strftime('%H:%M:%S')}] Verification FAILED: Detected '{detected_name}' != '{current_driver_name}'. Lock out.")
                else: # Initial recognition
                    recognition_passed = True

            if recognition_passed:
                if not driver_locked_in: print(f"[{time.strftime('%H:%M:%S')}] Driver locked in: {detected_name} (ID: {detected_id})")
                elif current_time - last_verification_time >= config.VERIFICATION_INTERVAL: print(f"[{time.strftime('%H:%M:%S')}] Re-verification OK for {detected_name}.")

                driver_authorized = True; driver_locked_in = True
                last_verification_time = current_time
                current_driver_id = detected_id; current_driver_name = detected_name
                status_text = f"Driver (Locked): {current_driver_name}"
                status_color = (0, 255, 0)
                if face_box: cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), status_color, 2)
            else: # Recognition failed or wrong driver during re-check
                if driver_locked_in: print(f"[{time.strftime('%H:%M:%S')}] Verification FAILED for {current_driver_name}. Locking out.")
                driver_authorized = False; driver_locked_in = False
                current_driver_id = None; current_driver_name = None
                last_verification_time = 0.0
                status_text = status_if_fail; status_color = (0, 0, 255)
                if DROWSINESS_ALERT: arduino_handler.send('0'); DROWSINESS_ALERT = False # Use handler
                COUNTER = 0

        # Display driver status
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # --- Stage 2: Drowsiness Detection ---
        if driver_authorized:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make MediaPipe processing optional or handle errors
            try:
                 results = face_mesh.process(image_rgb)
            except Exception as e:
                 print(f"Error processing frame with MediaPipe: {e}")
                 results = None # Ensure results is None on error


            ear = 1.0; eye_status = "Open"
            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_eye_lm, right_eye_lm = [], []
                valid_landmarks = True
                try:
                    for idx in config.LEFT_EYE_INDICES: left_eye_lm.append((int(face_landmarks.landmark[idx].x * image_width), int(face_landmarks.landmark[idx].y * image_height)))
                    for idx in config.RIGHT_EYE_INDICES: right_eye_lm.append((int(face_landmarks.landmark[idx].x * image_width), int(face_landmarks.landmark[idx].y * image_height)))
                    if len(left_eye_lm) != 6 or len(right_eye_lm) != 6: valid_landmarks = False
                except (IndexError, TypeError): valid_landmarks = False; eye_status = "Error: Landmarks"

                if valid_landmarks:
                    ear = (calculate_ear(left_eye_lm) + calculate_ear(right_eye_lm)) / 2.0
                    if ear < config.EYE_AR_THRESH:
                        COUNTER += 1; eye_status = "Closed"
                        if COUNTER >= config.EYE_CLOSED_THRESH and not DROWSINESS_ALERT:
                            DROWSINESS_ALERT = True; TOTAL_ALERTS += 1
                            arduino_handler.send('1') # Use handler
                            print(f"[{time.strftime('%H:%M:%S')}] ALERT: Drowsiness detected for {current_driver_name}!")
                    else:
                        eye_status = "Open"; COUNTER = 0
                        if DROWSINESS_ALERT:
                            DROWSINESS_ALERT = False; arduino_handler.send('0'); print(f"[{time.strftime('%H:%M:%S')}] Alert deactivated.")
                else: # Handle case where landmark extraction failed
                    ear = 1.0; eye_status = "Error: Landmarks"
                    COUNTER = 0
                    if DROWSINESS_ALERT: DROWSINESS_ALERT = False; arduino_handler.send('0')

            else: # No face landmarks detected by MediaPipe
                eye_status = "Open (No Landmarks)"; ear = 1.0; COUNTER = 0
                if DROWSINESS_ALERT: DROWSINESS_ALERT = False; arduino_handler.send('0')

            # Display Drowsiness Info
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eyes: {eye_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            alert_color = (0, 0, 255) if DROWSINESS_ALERT else (0, 255, 0)
            alert_text = "WAKE UP!" if DROWSINESS_ALERT else "Alert: Off"
            cv2.putText(frame, alert_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            cv2.putText(frame, f"Total Alerts: {TOTAL_ALERTS}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # --- Stage 3: Display Frame ---
        cv2.imshow('Drowsiness Detection & Driver Recognition', frame)

        # --- Stage 4: Exit ---
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Exit requested.")
            break

    # --- Cleanup ---
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    if face_mesh: face_mesh.close()
    # Arduino disconnect is handled in the main block's finally clause

# --- Main Execution ---
if __name__ == "__main__":
    print("--- System Start ---")
    # Initialize components
    init_database() # Ensure DB tables exist
    detector_net, embedder_net = load_face_models()
    arduino = ArduinoComm() # Create Arduino handler instance

    # Keep track of loaded state for cleanup
    models_loaded = detector_net is not None and embedder_net is not None
    arduino_connected_initially = False

    try:
        if models_loaded:
            # Run UI only if models loaded (UI needs models for encoding)
            print("\nLaunching Driver Manager UI...")
            print("Close the UI window when finished managing drivers to start detection.")
            run_driver_manager_ui(detector_net, embedder_net) # Pass models to UI
            print("Driver Manager UI closed.")

            # Load encodings AFTER UI is closed
            known_encodings = load_all_driver_encodings()
            if not known_encodings:
                 print("Warning: No known driver encodings loaded. Recognition may not work.")

            # Attempt to connect Arduino AFTER UI
            arduino.connect()
            arduino_connected_initially = arduino.is_connected() # Store status after connect attempt

            # Run main detection loop
            run_drowsiness_detection(arduino, detector_net, embedder_net, known_encodings)

        else:
             print("\nCritical Error: Face models failed to load. Cannot proceed.")
             print("Please ensure model files are correctly placed in the 'models' directory.")

    except Exception as main_exception:
        print(f"\n--- An unexpected error occurred in the main execution: ---")
        import traceback
        traceback.print_exc() # Print detailed traceback
        print(f"Error details: {main_exception}")
        print("-----------------------------------------------------------")

    finally:
        # Ensure Arduino disconnect is attempted regardless of errors
        print("--- System Shutdown ---")
        if arduino and arduino.is_connected():
            arduino.disconnect()
        elif arduino_connected_initially: # Log if initial connect worked but failed later
             print("Arduino was connected but may have encountered an error.")
        else:
             print("Arduino was not connected or connection failed.")
        print("Cleanup complete.")