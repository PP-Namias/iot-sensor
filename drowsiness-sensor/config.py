# config.py
import os

# --- General Settings ---
IMAGE_DIR = "images" # Directory to store driver images
DB_FILE = "drivers.db"
MODELS_DIR = "models" # Base directory for models

# --- Arduino Settings ---
ARDUINO_PORT = 'COM8' # <<<--- CHANGE THIS to your Arduino port (or None to disable)
ARDUINO_BAUDRATE = 9600
ARDUINO_TIMEOUT = 1

# --- Drowsiness Detection Settings ---
# Increasing threshold to make eye closing detection less sensitive
EYE_AR_THRESH = 0.20  # Lower value means eyes need to be more closed to trigger
EYE_CLOSED_THRESH = 20 # Increased number of consecutive frames for alert

# MediaPipe Face Mesh Settings
FACE_MESH_MAX_FACES = 1
FACE_MESH_REFINE_LANDMARKS = True
FACE_MESH_MIN_DETECTION_CONF = 0.5
FACE_MESH_MIN_TRACKING_CONF = 0.5

# Eye landmarks indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Facial Recognition Settings ---
# Model file paths (relative to MODELS_DIR)
_FACE_DETECTOR_PROTOTXT_REL = "deploy.prototxt"
_FACE_DETECTOR_MODEL_REL = "res10_300x300_ssd_iter_140000.caffemodel"
_FACE_EMBEDDING_MODEL_REL = "openface_nn4.small2.v1.t7"

# Construct full paths (handle potential errors if MODELS_DIR doesn't exist initially)
try:
    if not os.path.exists(MODELS_DIR):
        print(f"Warning: Models directory '{MODELS_DIR}' not found. Creating it.")
        os.makedirs(MODELS_DIR) # Create models dir if needed

    FACE_DETECTOR_PROTOTXT = os.path.join(MODELS_DIR, _FACE_DETECTOR_PROTOTXT_REL)
    FACE_DETECTOR_MODEL = os.path.join(MODELS_DIR, _FACE_DETECTOR_MODEL_REL)
    FACE_EMBEDDING_MODEL = os.path.join(MODELS_DIR, _FACE_EMBEDDING_MODEL_REL)
except Exception as e:
     print(f"Error setting up model paths in config.py: {e}")
     # Set to None or handle appropriately if paths can't be determined
     FACE_DETECTOR_PROTOTXT = None
     FACE_DETECTOR_MODEL = None
     FACE_EMBEDDING_MODEL = None

# FACE RECOGNITION CHANGES
# Lower detection confidence to catch more potential faces
FACE_DETECTION_CONFIDENCE = 0.5  # Was 0.6

# Increase recognition threshold to be more lenient with matching
FACE_RECOGNITION_THRESHOLD = 1.0  # Was 0.6 (higher value = more lenient matching)

# --- Lock-in Feature Settings ---
# Implementing the face locking feature you requested
FACE_LOCK_DURATION = 60.0  # Lock onto a face for 60 seconds (1 minute)
FACE_LOCK_REQUIRED_TIME = 2.0  # Seconds of continuous detection before locking

# Reduce verification interval to ensure we catch the face quickly
VERIFICATION_INTERVAL = 120.0  # Re-verify driver every 1 second