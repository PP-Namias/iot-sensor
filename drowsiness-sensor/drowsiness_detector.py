# drowsiness_detector.py
import math
import numpy as np # Might be needed if dist uses numpy points, but math.sqrt is fine

def is_eye_closed(ear, threshold=0.20):
    """
    Determine if an eye is closed based on EAR value.
    Typical thresholds: 0.15-0.25 (lower = more closed)
    """
    return ear < threshold

def detect_drowsiness(left_ear, right_ear, closed_duration, threshold=0.20, drowsy_time=2.0):
    """
    Detect drowsiness based on both eyes being closed for a duration.
    closed_duration: How long eyes have been closed in seconds
    drowsy_time: How many seconds of closed eyes indicates drowsiness
    """
    avg_ear = (left_ear + right_ear) / 2.0
    if avg_ear < threshold:
        # Eyes are closed, increment duration
        return True, closed_duration + 1/30.0  # Assuming 30 FPS
    else:
        # Eyes are open, reset duration
        return False, 0.0

def face_tracker(recognized_driver_id, timestamp, tracking_dict, lock_duration=60):
    """
    Track identified faces and lock onto them for specified duration.
    
    Args:
        recognized_driver_id: ID of recognized driver or None
        timestamp: Current timestamp
        tracking_dict: Dictionary to store tracking info {driver_id: (start_time, locked)}
        lock_duration: How long to lock onto a face in seconds
        
    Returns:
        driver_id: The ID to use (either new or locked)
        is_locked: Whether we're in a locked state
    """
    # No face detected
    if recognized_driver_id is None:
        return None, False
        
    # New face detected
    if recognized_driver_id not in tracking_dict:
        tracking_dict[recognized_driver_id] = (timestamp, False)
        return recognized_driver_id, False
        
    # Existing face
    start_time, locked = tracking_dict[recognized_driver_id]
    
    # Check if we should lock this face
    if not locked and (timestamp - start_time) >= 5:  # Lock after 5 seconds of consistent detection
        tracking_dict[recognized_driver_id] = (start_time, True)
        return recognized_driver_id, True
        
    # Check if we should unlock (recalibrate)
    if locked and (timestamp - start_time) >= lock_duration:
        # Reset tracking for this face
        tracking_dict[recognized_driver_id] = (timestamp, False)
        return recognized_driver_id, False
        
    # Continue with locked state
    return recognized_driver_id, locked

def dist(point1, point2):
    """Calculate Euclidean distance between two points."""
    # Ensure points are tuples/lists with at least 2 elements
    if len(point1) < 2 or len(point2) < 2:
        print("Warning: Invalid points received in dist function.")
        return 0.0 # Avoid error, return 0 distance
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except TypeError:
        print(f"Warning: TypeError calculating distance between {point1} and {point2}.")
        return 0.0

def calculate_ear(eye_landmarks):
    """Calculate the eye aspect ratio given the landmarks of an eye"""
    if eye_landmarks is None or len(eye_landmarks) != 6:
        # print("Warning: Invalid eye landmarks for EAR calculation.")
        return 1.0 # Assume open if landmarks are bad

    # Vertical eye landmarks P2-P6, P3-P5
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])

    # Horizontal eye landmark P1-P4
    C = dist(eye_landmarks[0], eye_landmarks[3])

    # Avoid division by zero
    if C < 1e-3: # Use a small threshold instead of exact zero
        return 1.0 # Return high EAR if horizontal distance is negligible

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear