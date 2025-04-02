import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import time
import sqlite3
import os
import PySimpleGUI as sg
# import threading # Not used in sequential execution

# --- Configuration ---
# Arduino Settings
ARDUINO_PORT = 'COM8' # <<<--- CHANGE THIS to your Arduino port
ARDUINO_BAUDRATE = 9600
ARDUINO_TIMEOUT = 1

# Drowsiness Detection Settings
EYE_AR_THRESH = 0.25  # Threshold for determining if the eye is closed
EYE_CLOSED_THRESH = 15 # Number of consecutive frames eyes must be closed
FACE_MESH_MAX_FACES = 1
FACE_MESH_REFINE_LANDMARKS = True
FACE_MESH_MIN_DETECTION_CONF = 0.5
FACE_MESH_MIN_TRACKING_CONF = 0.5

# Facial Recognition Settings
FACE_DETECTOR_PROTOTXT = "models/deploy.prototxt"
FACE_DETECTOR_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
FACE_EMBEDDING_MODEL = "models/openface_nn4.small2.v1.t7"
FACE_DETECTION_CONFIDENCE = 0.6 # Min confidence for DNN face detection
FACE_RECOGNITION_THRESHOLD = 0.6 # Max Euclidean distance for face match
IMAGE_DIR = "images" # Directory to store driver images
DB_FILE = "drivers.db"

# Eye landmarks indices for MediaPipe Face Mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Global Variables ---
arduino_connected = False
arduino = None
known_driver_encodings = [] # List to store [(driver_id, name, encoding), ...]
face_detector_net = None
face_embedder_net = None

# --- Database Management ---
# (Database functions remain unchanged - init_database, add_driver_db, search_driver_db, delete_driver_db, load_all_driver_encodings)
def init_database():
    """Initializes the SQLite database and creates the drivers table if it doesn't exist."""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def add_driver_db(driver_id, name, image_path, encoding):
    """Adds a new driver to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        # Serialize numpy array encoding to bytes
        encoding_blob = encoding.tobytes()
        cursor.execute(
            "INSERT INTO drivers (driver_id, name, image_path, encoding) VALUES (?, ?, ?, ?)",
            (driver_id, name, image_path, encoding_blob)
        )
        conn.commit()
        print(f"Driver {driver_id} added successfully.")
        return True
    except sqlite3.IntegrityError:
        print(f"Error: Driver ID '{driver_id}' already exists.")
        return False
    except Exception as e:
        print(f"Database Error adding driver: {e}")
        return False
    finally:
        conn.close()

def search_driver_db(driver_id):
    """Searches for a driver by ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT driver_id, name, image_path FROM drivers WHERE driver_id = ?", (driver_id,))
        result = cursor.fetchone()
        return result # Returns (driver_id, name, image_path) or None
    except Exception as e:
        print(f"Database Error searching driver: {e}")
        return None
    finally:
        conn.close()

def delete_driver_db(driver_id):
    """Deletes a driver by ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        # Optionally, delete the associated image file
        cursor.execute("SELECT image_path FROM drivers WHERE driver_id = ?", (driver_id,))
        result = cursor.fetchone()
        if result and os.path.exists(result[0]):
             try:
                 os.remove(result[0])
                 print(f"Deleted image file: {result[0]}")
             except OSError as e:
                 print(f"Error deleting image file {result[0]}: {e}")

        cursor.execute("DELETE FROM drivers WHERE driver_id = ?", (driver_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Driver {driver_id} deleted successfully.")
            return True
        else:
            print(f"Driver ID '{driver_id}' not found.")
            return False
    except Exception as e:
        print(f"Database Error deleting driver: {e}")
        return False
    finally:
        conn.close()

def load_all_driver_encodings():
    """Loads all driver IDs, names, and encodings from the database into memory."""
    global known_driver_encodings
    known_driver_encodings = []
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT driver_id, name, encoding FROM drivers")
        rows = cursor.fetchall()
        for row in rows:
            driver_id, name, encoding_blob = row
            # Deserialize blob back to numpy array (assuming float32, adjust if needed)
            encoding = np.frombuffer(encoding_blob, dtype=np.float32)
            known_driver_encodings.append((driver_id, name, encoding))
        print(f"Loaded {len(known_driver_encodings)} driver encodings from database.")
    except Exception as e:
        print(f"Database Error loading encodings: {e}")
    finally:
        conn.close()


# --- Facial Recognition ---
# (Facial recognition functions remain unchanged - load_face_models, get_face_encoding_from_image, detect_and_recognize_face)
def load_face_models():
    """Loads the DNN face detector and embedder models."""
    global face_detector_net, face_embedder_net
    try:
        print("Loading face detector model...")
        face_detector_net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_PROTOTXT, FACE_DETECTOR_MODEL)
        print("Loading face embedding model...")
        face_embedder_net = cv2.dnn.readNetFromTorch(FACE_EMBEDDING_MODEL)
        print("Face models loaded successfully.")
        return True
    except cv2.error as e:
        print(f"Error loading DNN models: {e}")
        print("Ensure model files are in the 'models' directory.")
        return False

def get_face_encoding_from_image(image_path):
    """Detects face in an image file and computes its embedding."""
    if not face_detector_net or not face_embedder_net:
        print("Error: Face models not loaded.")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # ---> ADDED: Robust check for empty/invalid image read <---
    if img.size == 0:
        print(f"Error: Image loaded from {image_path} is empty or invalid.")
        return None
    # ---> END ADDED CHECK <---

    (h, w) = img.shape[:2]
    if h == 0 or w == 0:
        print(f"Error: Image {image_path} has zero dimensions.")
        return None

    # Detect face using DNN detector
    try:
        resized_img = cv2.resize(img, (300, 300))
        if resized_img.size == 0:
            print("Error: Resized image is empty.")
            return None
        blob = cv2.dnn.blobFromImage(resized_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        if blob.size == 0:
            print("Error: Generated blob is empty for image detection.")
            return None
        face_detector_net.setInput(blob)
        detections = face_detector_net.forward()
    except cv2.error as e:
        print(f"OpenCV Error during face detection in image {image_path}: {e}")
        return None
    except Exception as e:
         print(f"Unexpected error during face detection in image {image_path}: {e}")
         return None


    # Find the detection with the highest confidence
    best_detection_idx = -1
    max_confidence = 0.0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > FACE_DETECTION_CONFIDENCE and confidence > max_confidence:
             max_confidence = confidence
             best_detection_idx = i

    if best_detection_idx == -1:
        print(f"No face detected with sufficient confidence in {image_path}")
        return None

    # Compute bounding box
    box = detections[0, 0, best_detection_idx, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # Ensure box is within image bounds
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(w - 1, endX), min(h - 1, endY)

    if startX >= endX or startY >= endY:
        print(f"Invalid face bounding box found in {image_path}.")
        return None

    # Extract face ROI
    face = img[startY:endY, startX:endX]
    if face.size == 0:
        print(f"Could not extract face ROI from {image_path}. Box: {(startX, startY, endX, endY)}")
        return None

    # Compute face embedding using OpenFace model
    try:
        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        if face_blob.size == 0:
             print("Error: Generated blob is empty for face embedding.")
             return None
        face_embedder_net.setInput(face_blob)
        vec = face_embedder_net.forward()
        return vec.flatten() # Return the 128-d embedding vector
    except cv2.error as e:
        print(f"Error computing embedding for {image_path}: {e}")
        return None
    except Exception as e:
         print(f"Unexpected error during face embedding for {image_path}: {e}")
         return None

def detect_and_recognize_face(frame, known_encodings):
    """Detects faces in a frame and tries to recognize them against known encodings."""
    # Added robust frame check in V3
    if frame is None or frame.size == 0:
        # print("Debug: Received empty frame in detect_and_recognize_face.") # Optional debug
        return None, None, None

    if not face_detector_net or not face_embedder_net:
        # print("Debug: Face models not loaded.") # Optional debug
        return None, None, None

    recognized_driver_id = None
    recognized_driver_name = None
    face_box = None
    min_dist = float('inf')
    detections = None # Initialize detections

    try:
        (h, w) = frame.shape[:2]
        if h == 0 or w == 0:
             # print("Debug: Frame has zero dimensions.") # Optional debug
             return None, None, None

        # Create blob (ensure frame is resized correctly first)
        resized_frame = cv2.resize(frame, (300, 300))
        if resized_frame.size == 0:
            # print("Debug: Resized frame is empty.") # Optional debug
            return None, None, None

        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        if blob.size == 0:
             # print("Debug: Generated blob is empty.") # Optional debug
             return None, None, None

        face_detector_net.setInput(blob)
        detections = face_detector_net.forward()

    except cv2.error as e:
        # Reduce noise by commenting out frequent errors if needed
        # print(f"OpenCV Error during blob creation/forward pass: {e}")
        return None, None, None
    except Exception as e:
         print(f"Unexpected error during blob creation/forward pass: {e}")
         return None, None, None

    # Ensure detections is valid before proceeding
    if detections is None or detections.shape[2] == 0:
         # print("Debug: No detections from face detector.") # Optional debug
         return None, None, None


    # Iterate through detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > FACE_DETECTION_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure box is valid
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            if startX >= endX or startY >= endY: continue

            face = frame[startY:endY, startX:endX]
            if face.size == 0: continue

            # Compute embedding for the detected face
            try:
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                if face_blob.size == 0: continue # Skip if embedding blob is empty

                face_embedder_net.setInput(face_blob)
                detected_encoding = face_embedder_net.forward().flatten()
            except cv2.error:
                continue # Skip if embedding fails
            except Exception as e:
                 print(f"Unexpected error during face embedding in loop: {e}")
                 continue

            # Compare with known encodings
            for driver_id, name, known_encoding in known_encodings:
                # Calculate Euclidean distance
                # Ensure encodings are valid numpy arrays before norm calculation
                if not isinstance(known_encoding, np.ndarray) or not isinstance(detected_encoding, np.ndarray):
                    continue
                distance = np.linalg.norm(known_encoding - detected_encoding)

                # Check if it's the best match so far and below threshold
                if distance < FACE_RECOGNITION_THRESHOLD and distance < min_dist:
                    min_dist = distance
                    recognized_driver_id = driver_id
                    recognized_driver_name = name
                    face_box = (startX, startY, endX, endY) # Store the box of the recognized face

            # If a match was found in this iteration, lock onto the first good match
            if recognized_driver_id is not None:
                break

    return recognized_driver_id, recognized_driver_name, face_box


# --- Drowsiness Detection Helpers (from original code) ---
# (calculate_ear, dist functions remain unchanged)
def calculate_ear(eye_landmarks):
    """Calculate the eye aspect ratio given the landmarks of an eye"""
    if len(eye_landmarks) != 6: return 1.0
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])
    C = dist(eye_landmarks[0], eye_landmarks[3])
    if C < 1e-3: return 1.0
    ear = (A + B) / (2.0 * C)
    return ear

def dist(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def send_to_arduino(command):
    """Send command to Arduino: '1' to turn on buzzer, '0' to turn off"""
    if arduino_connected and arduino:
        try:
            arduino.write(command.encode())
        except serial.SerialException as e:
            print(f"Arduino write error: {e}")


# --- User Interface (PySimpleGUI) ---
# (run_driver_manager_ui function remains unchanged)
def run_driver_manager_ui():
    """Runs the PySimpleGUI interface for managing drivers."""
    sg.theme('SystemDefault') # Or choose another theme

    layout = [
        [sg.Text('Driver Manager', font=('Helvetica', 16))],
        [sg.Text('Driver ID/Name:', size=(15,1)), sg.InputText(key='-DRIVER_ID-')],
        [sg.Text('Image File:', size=(15,1)), sg.Input(key='-IMAGE_PATH-', enable_events=True, readonly=True),
         sg.FileBrowse(file_types=(("Image Files", "*.jpg;*.png"),))],
        [sg.Button('Add Driver'), sg.Button('Search Driver'), sg.Button('Delete Driver')],
        [sg.Text('Status:', size=(15,1)), sg.Text('', size=(40,1), key='-STATUS-')],
        [sg.Multiline(size=(60, 10), key='-OUTPUT-', disabled=True, autoscroll=True)],
        [sg.Button('Exit Manager')]
    ]

    window = sg.Window('Truck Driver Management', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit Manager':
            break

        driver_id = values['-DRIVER_ID-'].strip()
        image_path = values['-IMAGE_PATH-']
        window['-STATUS-'].update('') # Clear status

        if event == 'Add Driver':
            if not driver_id:
                window['-STATUS-'].update('Error: Driver ID/Name cannot be empty.')
                continue
            if not image_path or not os.path.exists(image_path):
                window['-STATUS-'].update('Error: Please select a valid image file.')
                continue
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                 window['-STATUS-'].update('Error: Only JPG or PNG images are supported.')
                 continue

            # Create a unique filename in the images directory
            _, ext = os.path.splitext(image_path)
            safe_driver_id = "".join(c for c in driver_id if c.isalnum() or c in (' ', '_')).rstrip()
            new_filename = f"{safe_driver_id}_{int(time.time())}{ext}"
            destination_path = os.path.join(IMAGE_DIR, new_filename)

            try:
                # Copy image to storage
                import shutil
                shutil.copy(image_path, destination_path)

                # Compute encoding
                window['-STATUS-'].update('Processing image for encoding...')
                window.refresh() # Update UI to show status
                encoding = get_face_encoding_from_image(destination_path)

                if encoding is not None:
                    # Add to database (using driver_id as both ID and Name for simplicity here)
                    if add_driver_db(driver_id, driver_id, destination_path, encoding):
                        window['-OUTPUT-'].print(f"Added: ID={driver_id}, Image={destination_path}")
                        window['-STATUS-'].update('Driver added successfully!')
                        # Clear fields
                        window['-DRIVER_ID-'].update('')
                        window['-IMAGE_PATH-'].update('')
                    else:
                        window['-STATUS-'].update('Error adding driver (check console/output).')
                        os.remove(destination_path) # Clean up copied image if DB add failed
                else:
                    window['-STATUS-'].update('Error: Could not compute face encoding from image.')
                    os.remove(destination_path) # Clean up failed image

            except Exception as e:
                window['-STATUS-'].update(f'Error during add: {e}')
                if os.path.exists(destination_path):
                    try:
                        os.remove(destination_path)
                    except OSError:
                         pass # Ignore error if file couldn't be removed

        elif event == 'Search Driver':
            if not driver_id:
                window['-STATUS-'].update('Error: Enter Driver ID/Name to search.')
                continue

            result = search_driver_db(driver_id)
            if result:
                found_id, found_name, found_path = result
                window['-OUTPUT-'].print(f"Found: ID={found_id}, Name={found_name}, Image Path={found_path}")
                window['-STATUS-'].update('Driver found.')
                # Optionally display image if you add an sg.Image element
            else:
                window['-OUTPUT-'].print(f"Driver ID '{driver_id}' not found.")
                window['-STATUS-'].update('Driver not found.')

        elif event == 'Delete Driver':
            if not driver_id:
                window['-STATUS-'].update('Error: Enter Driver ID/Name to delete.')
                continue

            # Confirmation dialog
            confirm = sg.popup_yes_no(f'Are you sure you want to delete driver "{driver_id}"?', title='Confirm Deletion')
            if confirm == 'Yes':
                if delete_driver_db(driver_id):
                    window['-OUTPUT-'].print(f"Deleted driver: {driver_id}")
                    window['-STATUS-'].update('Driver deleted.')
                    window['-DRIVER_ID-'].update('') # Clear ID field after delete
                else:
                    window['-STATUS-'].update('Error deleting driver (check console/output).')

    window.close()


# --- (Keep all imports and other functions as they are) ---
# ... (Database, Facial Rec, Helpers, UI functions) ...

# --- Main Application Logic ---
def run_drowsiness_detection():
    """Runs the main drowsiness detection loop with facial recognition and periodic re-verification."""
    global arduino_connected, arduino # Allow modification

    # --- Initialization ---
    print("Initializing Drowsiness Detection System...")

    # 1. Initialize Arduino (Keep existing try-except block)
    try:
        time.sleep(0.5)
        arduino = serial.Serial(port=ARDUINO_PORT, baudrate=ARDUINO_BAUDRATE, timeout=ARDUINO_TIMEOUT)
        arduino_connected = True
        print(f"Attempting to connect to Arduino on {ARDUINO_PORT}...")
        time.sleep(2.5) # Increased delay
        send_to_arduino('0')
        time.sleep(0.1)
        print("Successfully connected to Arduino.")
    except serial.SerialException as e:
        arduino_connected = False
        print(f"Failed to connect to Arduino on {ARDUINO_PORT}: {e}. Running visual mode only.")
        arduino = None
    except Exception as e:
        arduino_connected = False
        print(f"An unexpected error occurred during Arduino initialization: {e}")
        arduino = None

    # 2. Initialize MediaPipe Face Mesh (Keep as is)
    print("Initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=FACE_MESH_MAX_FACES,
        refine_landmarks=FACE_MESH_REFINE_LANDMARKS,
        min_detection_confidence=FACE_MESH_MIN_DETECTION_CONF,
        min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONF
    )

    # 3. Load Driver Encodings from DB (Keep as is)
    print("Loading driver encodings...")
    load_all_driver_encodings()
    if not known_driver_encodings:
         print("Warning: No driver encodings found. Recognition will always fail.")

    # 4. Initialize Video Capture (Keep as is)
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        if arduino_connected and arduino: arduino.close()
        if 'face_mesh' in locals() and face_mesh: face_mesh.close()
        return

    # --- State Variables for Main Loop ---
    COUNTER = 0
    TOTAL_ALERTS = 0
    DROWSINESS_ALERT = False
    current_driver_id = None
    current_driver_name = None
    driver_authorized = False
    driver_locked_in = False
    # ---> ADDED: Variables for periodic check <---
    last_verification_time = 0.0  # Timestamp of the last successful verification
    VERIFICATION_INTERVAL = 5.0  # Check every 5 seconds
    # ---> END ADDED <---

    print("Starting detection loop...")
    # --- Main Loop ---
    while cap.isOpened():
        # ---> Get current time at the start of the loop <---
        current_time = time.time()

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape
        if image_height == 0 or image_width == 0: continue

        # ---> MODIFIED: Stage 1 with Periodic Re-verification Logic <---
        perform_recognition_this_frame = False
        status_text = ""
        status_color = (0, 0, 0)
        status_if_recognition_fails = "Unauthorized Driver / No Face" # Default status if initial check fails

        if not driver_locked_in:
            # If not locked in, always attempt recognition
            perform_recognition_this_frame = True
        else:
            # If locked in, check if verification interval has passed
            if current_time - last_verification_time >= VERIFICATION_INTERVAL:
                perform_recognition_this_frame = True
                status_if_recognition_fails = "RE-VERIFYING..." # Show this if re-check fails
                print(f"[{time.strftime('%H:%M:%S')}] Re-verifying driver: {current_driver_name}") # Log re-check attempt
            else:
                # Interval not passed, stay locked in without checking
                perform_recognition_this_frame = False
                driver_authorized = True # Maintain authorization
                status_text = f"Driver (Locked): {current_driver_name}"
                status_color = (0, 255, 0)

        # --- Perform Recognition if Required ---
        if perform_recognition_this_frame:
            detected_id, detected_name, face_box = detect_and_recognize_face(frame, known_driver_encodings)

            recognition_passed = False
            if detected_id is not None:
                if driver_locked_in:
                    # During re-verification, must match the locked-in driver
                    if detected_id == current_driver_id:
                        recognition_passed = True
                    else:
                        # Detected DIFFERENT driver during re-check
                        print(f"[{time.strftime('%H:%M:%S')}] Verification FAILED: Detected '{detected_name}' instead of '{current_driver_name}'. Locking out.")
                        recognition_passed = False # Explicitly false
                else:
                    # Initial recognition, any valid driver is okay
                    recognition_passed = True

            # --- Handle Recognition Result ---
            if recognition_passed:
                # --- Success: Lock-in or Maintain Lock ---
                if not driver_locked_in: # Log initial lock-in
                     print(f"[{time.strftime('%H:%M:%S')}] Driver locked in: {detected_name} (ID: {detected_id})")
                elif current_time - last_verification_time >= VERIFICATION_INTERVAL: # Log successful re-verification
                     print(f"[{time.strftime('%H:%M:%S')}] Re-verification successful for {detected_name}.")


                driver_authorized = True
                driver_locked_in = True
                last_verification_time = current_time # Update timestamp on success
                current_driver_id = detected_id    # Update driver info
                current_driver_name = detected_name
                status_text = f"Driver (Locked): {current_driver_name}"
                status_color = (0, 255, 0)
                if face_box: # Draw box on successful check
                    (startX, startY, endX, endY) = face_box
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            else:
                # --- Failure: Lock Out or Remain Unauthorized ---
                if driver_locked_in: # Only log lock-out if previously locked in
                    print(f"[{time.strftime('%H:%M:%S')}] Verification FAILED for {current_driver_name}. Locking out.")

                driver_authorized = False
                driver_locked_in = False # LOCK OUT
                current_driver_id = None
                current_driver_name = None
                last_verification_time = 0.0 # Reset timer

                status_text = status_if_recognition_fails # Show "Re-verifying..." or "Unauthorized..."
                status_color = (0, 0, 255) # Red

                # Reset drowsiness state on lock out/failure
                if DROWSINESS_ALERT:
                    send_to_arduino('0')
                    DROWSINESS_ALERT = False
                COUNTER = 0
        # ---> END MODIFIED Stage 1 <---


        # Display driver status on frame
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)


        # --- Stage 2: Drowsiness Detection (Only if Authorized) ---
        # (This section remains unchanged, it just depends on driver_authorized flag)
        if driver_authorized:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            ear = 1.0
            eye_status = "Open"
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_eye_landmarks, right_eye_landmarks = [], []
                try:
                    for idx in LEFT_EYE_INDICES:
                        lm = face_landmarks.landmark[idx]; x, y = int(lm.x * image_width), int(lm.y * image_height); left_eye_landmarks.append((x, y))
                    for idx in RIGHT_EYE_INDICES:
                        lm = face_landmarks.landmark[idx]; x, y = int(lm.x * image_width), int(lm.y * image_height); right_eye_landmarks.append((x, y))

                    if len(left_eye_landmarks) == 6 and len(right_eye_landmarks) == 6:
                        ear = (calculate_ear(left_eye_landmarks) + calculate_ear(right_eye_landmarks)) / 2.0
                    else: ear = 1.0

                except (IndexError, TypeError): ear = 1.0; eye_status = "Error: Landmarks"

                if ear < EYE_AR_THRESH:
                    COUNTER += 1; eye_status = "Closed"
                    if COUNTER >= EYE_CLOSED_THRESH and not DROWSINESS_ALERT:
                        DROWSINESS_ALERT = True; TOTAL_ALERTS += 1; send_to_arduino('1')
                        print(f"[{time.strftime('%H:%M:%S')}] ALERT: Drowsiness detected for {current_driver_name}!")
                else:
                    eye_status = "Open"; COUNTER = 0
                    if DROWSINESS_ALERT:
                        DROWSINESS_ALERT = False; send_to_arduino('0'); print(f"[{time.strftime('%H:%M:%S')}] Alert deactivated.")
            else: # No landmarks
                eye_status = "Open (No Landmarks)"; ear = 1.0; COUNTER = 0
                if DROWSINESS_ALERT: DROWSINESS_ALERT = False; send_to_arduino('0')

            # Display Drowsiness Info
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eyes: {eye_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            alert_color = (0, 0, 255) if DROWSINESS_ALERT else (0, 255, 0)
            alert_text = "WAKE UP!" if DROWSINESS_ALERT else "Alert: Off"
            cv2.putText(frame, alert_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            cv2.putText(frame, f"Total Alerts: {TOTAL_ALERTS}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        # --- Stage 3: Display Frame ---
        cv2.imshow('Drowsiness Detection and Driver Recognition', frame)

        # --- Stage 4: Exit Condition ---
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Exit requested.")
            break

    # --- Cleanup --- (Keep existing cleanup logic)
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    if 'face_mesh' in locals() and face_mesh: face_mesh.close()
    if arduino_connected and arduino:
        try:
            send_to_arduino('0')
            time.sleep(0.1)
            arduino.close()
            print("Arduino connection closed.")
        except Exception as e:
            print(f"Error closing Arduino connection: {e}")
    print("System shut down.")


# --- Main Execution --- (Keep as is)
if __name__ == "__main__":
    init_database()
    if not load_face_models(): exit()
    print("\nLaunching Driver Manager UI...")
    print("Close the UI window when finished managing drivers to start detection.")
    run_driver_manager_ui()
    print("Driver Manager UI closed.")
    print("\nStarting Drowsiness Detection System...")
    run_drowsiness_detection()