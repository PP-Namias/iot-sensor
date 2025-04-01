import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize variables for motion tracking
prev_landmarks = None
motion_threshold = 30  # Adjust as needed

def calculate_motion(prev_landmarks, current_landmarks):
    """Calculate the motion between two sets of landmarks"""
    if prev_landmarks is None or current_landmarks is None:
        return 0
    
    # Make sure we're comparing the same number of points
    min_points = min(len(prev_landmarks), len(current_landmarks))
    if min_points == 0:
        return 0
    
    total_motion = 0
    for i in range(min_points):
        # Calculate Euclidean distance between corresponding landmarks
        dx = prev_landmarks[i][0] - current_landmarks[i][0]
        dy = prev_landmarks[i][1] - current_landmarks[i][1]
        distance = math.sqrt(dx**2 + dy**2)
        total_motion += distance
    
    # Return average motion per point for more consistent readings
    return total_motion / min_points

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam.")
        continue
    
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    # Draw hand landmarks on the image
    image_height, image_width, _ = image.shape
    current_landmarks = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract landmark coordinates
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                current_landmarks.append((x, y))
    
    # Calculate motion only if we have current landmarks
    motion = 0
    if current_landmarks:
        motion = calculate_motion(prev_landmarks, current_landmarks)
        # Update previous landmarks only when we have valid current landmarks
        prev_landmarks = current_landmarks.copy()
    
    # Display motion value
    cv2.putText(
        image,
        f"Motion: {motion:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if motion > motion_threshold else (0, 0, 255),
        2
    )
    
    # Detect significant motion
    if motion > motion_threshold:
        cv2.putText(
            image,
            "Motion Detected!",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    # Display the image
    cv2.imshow('Hand Motion Detection', image)
    
    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()