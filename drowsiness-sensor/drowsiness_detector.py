# drowsiness_detector.py
import math
import numpy as np # Might be needed if dist uses numpy points, but math.sqrt is fine

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