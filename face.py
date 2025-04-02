import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import time

# Initialize Arduino connection
# Change 'COM3' to the correct port for your Arduino
try: 
    arduino = serial.Serial(port='COM8', baudrate=9600, timeout=1)
    arduino_connected = True
    print("Successfully connected to Arduino")
    time.sleep(2)  # Give Arduino time to reset
except:
    arduino_connected = False
    print("Failed to connect to Arduino. Running in visual mode only.")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks indices for MediaPipe Face Mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Constants for the eye detection
EYE_AR_THRESH = 0.25  # Threshold for determining if the eye is closed
EYE_CLOSED_THRESH = 10  # Number of consecutive frames eyes must be closed to trigger alert
DROWSINESS_ALERT = False  # Tracks if an alert is currently active

def calculate_ear(eye_landmarks):
    """Calculate the eye aspect ratio given the landmarks of an eye"""
    # Vertical eye landmarks (top to bottom)
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal eye landmark (left to right)
    C = dist(eye_landmarks[0], eye_landmarks[3])
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def dist(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def send_to_arduino(command):
    """Send command to Arduino: '1' to turn on buzzer, '0' to turn off"""
    if arduino_connected:
        arduino.write(command.encode())

# Initialize counters
COUNTER = 0
TOTAL_ALERTS = 0
eye_status = "Open"

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam.")
        continue
    
    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect face landmarks
    results = face_mesh.process(image_rgb)
    
    # Draw face landmarks
    image_height, image_width, _ = image.shape
    
    # Default to eyes open if no face is detected
    if not results.multi_face_landmarks:
        if DROWSINESS_ALERT:
            send_to_arduino('0')  # Turn off buzzer
            DROWSINESS_ALERT = False
        COUNTER = 0
    
    for face_landmarks in (results.multi_face_landmarks or []):
        # Extract eye landmarks
        left_eye_landmarks = []
        right_eye_landmarks = []
        
        for idx in LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            left_eye_landmarks.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        
        for idx in RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            right_eye_landmarks.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        
        # Draw contours around eyes
        cv2.polylines(image, [np.array(left_eye_landmarks, dtype=np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(image, [np.array(right_eye_landmarks, dtype=np.int32)], True, (0, 255, 0), 1)
        
        # Calculate the eye aspect ratio for both eyes
        left_ear = calculate_ear(left_eye_landmarks)
        right_ear = calculate_ear(right_eye_landmarks)
        
        # Average the eye aspect ratio for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            eye_status = "Closed"
            
            # If eyes have been closed for too long, trigger alert
            if COUNTER >= EYE_CLOSED_THRESH and not DROWSINESS_ALERT:
                DROWSINESS_ALERT = True
                TOTAL_ALERTS += 1
                send_to_arduino('1')  # Turn on buzzer
                
        else:
            COUNTER = 0
            eye_status = "Open"
            if DROWSINESS_ALERT:
                DROWSINESS_ALERT = False
                send_to_arduino('0')  # Turn off buzzer
        
        # Display the eye aspect ratio and status on the frame
        cv2.putText(image, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Eyes: {eye_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display alert status
        alert_color = (0, 0, 255) if DROWSINESS_ALERT else (0, 255, 0)
        alert_text = "WAKE UP!" if DROWSINESS_ALERT else "Alert: Off"
        cv2.putText(image, alert_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        cv2.putText(image, f"Total Alerts: {TOTAL_ALERTS}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the image
    cv2.imshow('Drowsiness Detection', image)
    
    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        # Make sure to turn off the buzzer before exiting
        if arduino_connected and DROWSINESS_ALERT:
            send_to_arduino('0')
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
if arduino_connected:
    arduino.close()