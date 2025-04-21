# face_recognizer.py
import cv2
import numpy as np
import os
import config # Import configuration

def load_face_models():
    """Loads the DNN face detector and embedder models. Returns (detector, embedder) or (None, None)."""
    try:
        # Check if model files exist before attempting to load
        if not os.path.exists(config.FACE_DETECTOR_PROTOTXT) or \
           not os.path.exists(config.FACE_DETECTOR_MODEL) or \
           not os.path.exists(config.FACE_EMBEDDING_MODEL):
            print("Error: One or more DNN model files not found at specified paths:")
            print(f" - Prototxt: {config.FACE_DETECTOR_PROTOTXT} (Exists: {os.path.exists(config.FACE_DETECTOR_PROTOTXT)})")
            print(f" - CaffeModel: {config.FACE_DETECTOR_MODEL} (Exists: {os.path.exists(config.FACE_DETECTOR_MODEL)})")
            print(f" - TorchModel: {config.FACE_EMBEDDING_MODEL} (Exists: {os.path.exists(config.FACE_EMBEDDING_MODEL)})")
            return None, None

        print("Loading face detector model...")
        face_detector = cv2.dnn.readNetFromCaffe(config.FACE_DETECTOR_PROTOTXT, config.FACE_DETECTOR_MODEL)
        print("Loading face embedding model...")
        face_embedder = cv2.dnn.readNetFromTorch(config.FACE_EMBEDDING_MODEL)
        print("Face models loaded successfully.")
        return face_detector, face_embedder
    except cv2.error as e:
        print(f"OpenCV Error loading DNN models: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading models: {e}")
        return None, None

def get_face_encoding_from_image(image_path, face_detector, face_embedder):
    """Detects face in an image file and computes its embedding using loaded models."""
    if not face_detector or not face_embedder:
        print("Error: Face models not available for encoding.")
        return None

    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        print(f"Error: Could not read or invalid image {image_path}")
        return None

    (h, w) = img.shape[:2]
    if h == 0 or w == 0: return None # Invalid dimensions

    detections = None
    try:
        resized_img = cv2.resize(img, (300, 300))
        if resized_img.size == 0: return None
        blob = cv2.dnn.blobFromImage(resized_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        if blob.size == 0: return None

        face_detector.setInput(blob)
        detections = face_detector.forward()
    except Exception as e:
        print(f"Error during face detection for encoding ({image_path}): {e}")
        return None

    if detections is None: return None

    # Find best detection
    best_detection_idx = -1
    max_confidence = 0.0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > config.FACE_DETECTION_CONFIDENCE and confidence > max_confidence:
             max_confidence = confidence
             best_detection_idx = i

    if best_detection_idx == -1: return None # No face found

    # Extract face ROI
    box = detections[0, 0, best_detection_idx, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(w - 1, endX), min(h - 1, endY)
    if startX >= endX or startY >= endY: return None # Invalid box

    face = img[startY:endY, startX:endX]
    if face.size == 0: return None # Empty ROI

    # Compute embedding
    try:
        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        if face_blob.size == 0: return None
        face_embedder.setInput(face_blob)
        vec = face_embedder.forward()
        return vec.flatten()
    except Exception as e:
        print(f"Error computing embedding ({image_path}): {e}")
        return None

def detect_and_recognize_face(frame, face_detector, face_embedder, known_encodings):
    """
    Detects faces, computes embeddings, and compares against known encodings.
    Returns information about both identified and unidentified faces.
    """
    if frame is None or frame.size == 0 or not face_detector or not face_embedder:
        return None, None, None, False

    recognized_driver_id = None
    recognized_driver_name = None
    face_box = None
    is_unidentified = False
    min_dist = float('inf')
    detections = None

    try:
        (h, w) = frame.shape[:2]
        if h == 0 or w == 0: return None, None, None, False

        resized_frame = cv2.resize(frame, (300, 300))
        if resized_frame.size == 0: return None, None, None, False
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        if blob.size == 0: return None, None, None, False

        face_detector.setInput(blob)
        detections = face_detector.forward()

    except Exception as e:
        # print(f"Error during live face detection: {e}") # Can be noisy
        return None, None, None, False

    if detections is None or detections.shape[2] == 0:
         return None, None, None, False

    # Iterate through detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > config.FACE_DETECTION_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            if startX >= endX or startY >= endY: continue

            face = frame[startY:endY, startX:endX]
            if face.size == 0: continue

            # Compute embedding for the detected face
            detected_encoding = None
            try:
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                if face_blob.size == 0: continue
                face_embedder.setInput(face_blob)
                detected_encoding = face_embedder.forward().flatten()
            except Exception:
                continue # Skip if embedding fails

            if detected_encoding is None: continue

            # Compare with known encodings
            matched = False
            for driver_id, name, known_encoding in known_encodings:
                if not isinstance(known_encoding, np.ndarray): continue
                try:
                     distance = np.linalg.norm(known_encoding - detected_encoding)
                     if distance < config.FACE_RECOGNITION_THRESHOLD and distance < min_dist:
                        min_dist = distance
                        recognized_driver_id = driver_id
                        recognized_driver_name = name
                        face_box = (startX, startY, endX, endY)
                        matched = True
                except Exception as e:
                     print(f"Error calculating distance: {e}") # Should not happen often
            
            # If no match found but face detected, mark as unidentified
            if not matched:
                is_unidentified = True
                face_box = (startX, startY, endX, endY)
                # Only set as unidentified if we don't have a recognized driver
                if recognized_driver_id is None:
                    recognized_driver_id = "unidentified"
                    recognized_driver_name = "Unidentified Driver"

            # Lock onto the first good match found in this frame's detections
            if recognized_driver_id is not None:
                break # Exit detection loop once a match is found

    return recognized_driver_id, recognized_driver_name, face_box, is_unidentified
