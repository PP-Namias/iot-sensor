import cv2
import mediapipe as mp
import numpy as np
import time
import os
import psutil
import threading
import matplotlib.pyplot as plt
from collections import deque
import pygame  # For sound alerts

# Import project modules
import config
from database_manager import init_database, load_all_driver_encodings
from face_recognizer import load_face_models, detect_and_recognize_face
from drowsiness_detector import calculate_ear, dist  # Import EAR helpers
from arduino_comm import ArduinoComm
from ui_manager import run_driver_manager_ui
from logging_config import log_performance, log_event
import logging

# Initialize pygame mixer for sound alerts
pygame.mixer.init()

class PerformanceMonitor:
    """Monitors and records CPU and memory usage of the application."""
    
    def __init__(self, logging_interval=1.0, history_size=300):
        """Initialize performance monitor.
        
        Args:
            logging_interval: Time in seconds between measurements
            history_size: Number of measurements to keep in history
        """
        self.process = psutil.Process(os.getpid())
        self.logging_interval = logging_interval
        self.running = False
        self.monitor_thread = None
        
        # Performance metrics history
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # Frame processing metrics
        self.frame_times = deque(maxlen=history_size)
        self.last_frame_time = None
        self.frame_count = 0
        self.fps_history = deque(maxlen=history_size)
        
        # Peak values
        self.peak_cpu = 0
        self.peak_memory = 0
        self.min_fps = float('inf')
        self.max_fps = 0
        
    def start(self):
        """Start the monitoring thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Performance monitoring started")  # Keep console log
            log_event("Performance monitoring started")  # Add log entry
            
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
        print("Performance monitoring stopped")  # Keep console log
        log_event("Performance monitoring stopped")  # Add log entry
            
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.running:
            try:
                # Get CPU usage as percentage (across all cores)
                cpu_percent = self.process.cpu_percent()
                
                # Get memory usage in MB
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Update peak values
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # Record current time
                current_time = time.time()
                
                # Store measurements
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_mb)
                self.timestamp_history.append(current_time)
                
                # Log the performance metrics
                metrics = {
                    'cpu': cpu_percent,
                    'memory': memory_mb,
                    'fps': self.fps_history[-1] if self.fps_history else 0
                }
                log_performance(metrics)  # Log performance metrics
                
                time.sleep(self.logging_interval)
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")  # Keep console log
                log_event(f"Error in performance monitoring: {e}", level="error")  # Add log entry
                time.sleep(self.logging_interval)
    
    def record_frame_processed(self):
        """Record that a frame was processed."""
        current_time = time.time()
        self.frame_count += 1
        
        if self.last_frame_time is not None:
            # Calculate instantaneous FPS
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            # Calculate rolling FPS (over last ~1 second)
            if len(self.frame_times) >= 10:
                fps = len(self.frame_times) / sum(self.frame_times)
                self.fps_history.append(fps)
                self.min_fps = min(self.min_fps, fps) if fps > 0 else self.min_fps
                self.max_fps = max(self.max_fps, fps)
        
        self.last_frame_time = current_time
    
    def get_current_metrics(self):
        """Get the latest performance metrics."""
        current_cpu = self.cpu_history[-1] if self.cpu_history else 0
        current_memory = self.memory_history[-1] if self.memory_history else 0
        current_fps = self.fps_history[-1] if self.fps_history else 0
        
        return {
            'cpu': current_cpu,
            'memory': current_memory,
            'fps': current_fps
        }
    
    def get_summary(self):
        """Get a summary of performance metrics."""
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        return {
            'avg_cpu': avg_cpu,
            'peak_cpu': self.peak_cpu,
            'avg_memory': avg_memory,
            'peak_memory': self.peak_memory,
            'avg_fps': avg_fps,
            'min_fps': self.min_fps if self.min_fps != float('inf') else 0,
            'max_fps': self.max_fps,
            'total_frames': self.frame_count
        }
    
    def save_performance_graph(self, filename='performance_metrics.png'):
        """Save performance metrics graph to file."""
        if not self.cpu_history or not self.memory_history:
            print("No performance data to graph")
            return
        
        # Convert timestamps to relative time (0 = start)
        start_time = self.timestamp_history[0]
        relative_times = [(t - start_time) for t in self.timestamp_history]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # CPU usage plot
        ax1.plot(relative_times, self.cpu_history, 'b-')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Usage Over Time')
        ax1.grid(True)
        
        # Memory usage plot
        ax2.plot(relative_times, self.memory_history, 'r-')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Over Time')
        ax2.grid(True)
        
        # FPS plot
        if self.fps_history:
            fps_times = relative_times[-len(self.fps_history):]
            ax3.plot(fps_times, self.fps_history, 'g-')
            ax3.set_ylabel('FPS')
            ax3.set_title('Frames Per Second')
            ax3.grid(True)
        
        ax3.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Performance graph saved to {filename}")


def run_drowsiness_detection(arduino_handler, face_detector, face_embedder, known_encodings):
    """Runs the main drowsiness detection loop with facial recognition and periodic re-verification."""
    print("\nStarting Drowsiness Detection System Loop...")
    log_event("Starting Drowsiness Detection System Loop")  # Add log entry

    # Initialize performance monitoring
    perf_monitor = PerformanceMonitor(logging_interval=0.5)
    perf_monitor.start()

    # Initialize MediaPipe Face Mesh
    print("Initializing MediaPipe Face Mesh...")
    log_event("Initializing MediaPipe Face Mesh")  # Add log entry
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Custom drawing specs for better visualization
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
    
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
        perf_monitor.stop()
        return  # Cannot proceed without face mesh

    # Initialize Video Capture
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        log_event("Error: Could not open webcam", level="error")  # Add log entry
        if face_mesh: face_mesh.close()
        perf_monitor.stop()
        return

    # Load alert sound
    alert_sound_path = os.path.join(os.path.dirname(__file__), "alert.mp3")
    # If alert sound doesn't exist, create a default one
    if not os.path.exists(alert_sound_path):
        try:
            # Generate a simple beep sound using pygame
            pygame.mixer.Sound(np.sin(2*np.pi*np.arange(44100*1)/44100*440).astype(np.float32)).save(alert_sound_path)
            print(f"Created default alert sound at {alert_sound_path}")
        except Exception as e:
            print(f"Could not create default alert sound: {e}")
            alert_sound_path = None
    
    # Load the alert sound if available
    alert_sound = None
    try:
        if alert_sound_path and os.path.exists(alert_sound_path):
            alert_sound = pygame.mixer.Sound(alert_sound_path)
            print("Alert sound loaded successfully")
    except Exception as e:
        print(f"Error loading alert sound: {e}")
        alert_sound = None

    # --- State Variables for Main Loop ---
    frame_count = 0  # Optional: for frame-based logic or debugging
    COUNTER = 0
    TOTAL_ALERTS = 0
    DROWSINESS_ALERT = False
    current_driver_id = None
    current_driver_name = None
    driver_authorized = False
    driver_locked_in = False
    last_verification_time = 0.0
    is_unidentified_driver = False
    
    # For calculating time spent in each processing stage
    stage_times = {
        'frame_capture': 0.0,
        'face_recognition': 0.0,
        'drowsiness_detection': 0.0,
        'display': 0.0
    }

    print("Detection loop running... Press ESC in the video window to exit.")
    # --- Main Loop ---
    while cap.isOpened():
        loop_start_time = time.time()
        current_time = loop_start_time
        frame_count += 1

        # --- Stage 0: Frame Capture ---
        frame_start_time = time.time()
        success, frame = cap.read()
        if not success:
            # print("Warning: Failed to read frame.") # Can be noisy
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape
        if image_height == 0 or image_width == 0: continue  # Skip invalid frames
        
        stage_times['frame_capture'] = time.time() - frame_start_time

        # --- Stage 1: Face Recognition / Verification ---
        recog_start_time = time.time()
        perform_recognition = False
        status_text = ""
        status_color = (0, 0, 0)
        status_if_fail = "No Face Detected"

        if not driver_locked_in:
            perform_recognition = True
        else:
            if current_time - last_verification_time >= config.VERIFICATION_INTERVAL:
                perform_recognition = True
                status_if_fail = "RE-VERIFYING..."
            else:
                perform_recognition = False
                driver_authorized = True  # Maintain lock
                if is_unidentified_driver:
                    status_text = f"Driver (Unidentified)"
                    status_color = (255, 165, 0)  # Orange for unidentified
                else:
                    status_text = f"Driver (Locked): {current_driver_name}"
                    status_color = (0, 255, 0)  # Green for identified

        if perform_recognition:
            # Pass loaded models to the recognition function
            detected_id, detected_name, face_box, is_unidentified = detect_and_recognize_face(
                frame, face_detector, face_embedder, known_encodings
            )
            
            # Log appropriate message based on detection result
            if detected_id == "unidentified" and is_unidentified:
                log_event("Unidentified driver detected")
            elif detected_id is not None:
                log_event(f"Driver recognized: {detected_name} (ID: {detected_id})")
            else:
                log_event("No driver detected", level="warning")

            recognition_passed = False
            if detected_id is not None:
                if driver_locked_in and not is_unidentified_driver:  # Re-verification check for identified driver
                    if detected_id == current_driver_id: 
                        recognition_passed = True
                    else: 
                        print(f"[{time.strftime('%H:%M:%S')}] Verification FAILED: Detected '{detected_name}' != '{current_driver_name}'. Lock out.")
                else:  # Initial recognition or unidentified driver
                    recognition_passed = True

            if recognition_passed:
                if not driver_locked_in:
                    if detected_id == "unidentified":
                        print(f"[{time.strftime('%H:%M:%S')}] Unidentified driver detected and locked in")
                        is_unidentified_driver = True
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Driver locked in: {detected_name} (ID: {detected_id})")
                        is_unidentified_driver = False
                elif current_time - last_verification_time >= config.VERIFICATION_INTERVAL:
                    if detected_id == "unidentified":
                        print(f"[{time.strftime('%H:%M:%S')}] Re-verification OK for unidentified driver.")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Re-verification OK for {detected_name}.")

                driver_authorized = True
                driver_locked_in = True
                last_verification_time = current_time
                current_driver_id = detected_id
                current_driver_name = detected_name
                
                if is_unidentified_driver:
                    status_text = "Driver (Unidentified)"
                    status_color = (255, 165, 0)  # Orange for unidentified
                else:
                    status_text = f"Driver (Locked): {current_driver_name}"
                    status_color = (0, 255, 0)  # Green for identified
                
                if face_box: 
                    cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), status_color, 2)
            else:  # Recognition failed or wrong driver during re-check
                if driver_locked_in: 
                    print(f"[{time.strftime('%H:%M:%S')}] Verification FAILED. Locking out.")
                driver_authorized = False
                driver_locked_in = False
                current_driver_id = None
                current_driver_name = None
                is_unidentified_driver = False
                last_verification_time = 0.0
                status_text = status_if_fail
                status_color = (0, 0, 255)
                if DROWSINESS_ALERT: 
                    arduino_handler.send('0')
                    DROWSINESS_ALERT = False
                COUNTER = 0

        # Display driver status
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        stage_times['face_recognition'] = time.time() - recog_start_time

        # --- Stage 2: Drowsiness Detection ---
        drowsy_start_time = time.time()
        if driver_authorized:  # Apply drowsiness detection for both identified and unidentified drivers
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make MediaPipe processing optional or handle errors
            try:
                results = face_mesh.process(image_rgb)
            except Exception as e:
                print(f"Error processing frame with MediaPipe: {e}")
                results = None  # Ensure results is None on error

            ear = 1.0
            eye_status = "Open"
            
            # Draw face mesh landmarks if available
            if results and results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw the face contours
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Draw eyes specifically with different color
                    left_eye_lm, right_eye_lm = [], []
                    valid_landmarks = True
                    try:
                        # Extract and draw left eye landmarks
                        for idx in config.LEFT_EYE_INDICES: 
                            x = int(face_landmarks.landmark[idx].x * image_width)
                            y = int(face_landmarks.landmark[idx].y * image_height)
                            left_eye_lm.append((x, y))
                            # Draw eye landmark with larger circle and different color
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                        
                        # Extract and draw right eye landmarks
                        for idx in config.RIGHT_EYE_INDICES: 
                            x = int(face_landmarks.landmark[idx].x * image_width)
                            y = int(face_landmarks.landmark[idx].y * image_height)
                            right_eye_lm.append((x, y))
                            # Draw eye landmark with larger circle and different color
                            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                        
                        # Draw nose tip with different color
                        nose_tip_idx = 4  # MediaPipe nose tip landmark index
                        nose_x = int(face_landmarks.landmark[nose_tip_idx].x * image_width)
                        nose_y = int(face_landmarks.landmark[nose_tip_idx].y * image_height)
                        cv2.circle(frame, (nose_x, nose_y), 3, (0, 0, 255), -1)
                        
                        # Draw mouth landmarks with different color
                        mouth_indices = [61, 291, 0, 17]  # Some key mouth landmarks
                        for idx in mouth_indices:
                            x = int(face_landmarks.landmark[idx].x * image_width)
                            y = int(face_landmarks.landmark[idx].y * image_height)
                            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
                        
                        if len(left_eye_lm) != 6 or len(right_eye_lm) != 6: 
                            valid_landmarks = False
                    except (IndexError, TypeError): 
                        valid_landmarks = False
                        eye_status = "Error: Landmarks"

                    if valid_landmarks:
                        ear = (calculate_ear(left_eye_lm) + calculate_ear(right_eye_lm)) / 2.0
                        if ear < config.EYE_AR_THRESH:
                            COUNTER += 1
                            eye_status = "Closed"
                            if COUNTER >= config.EYE_CLOSED_THRESH and not DROWSINESS_ALERT:
                                DROWSINESS_ALERT = True
                                TOTAL_ALERTS += 1
                                arduino_handler.send('1')  # Use handler
                                
                                # Play alert sound if available
                                if alert_sound:
                                    alert_sound.play()
                                
                                driver_type = "unidentified driver" if is_unidentified_driver else current_driver_name
                                print(f"[{time.strftime('%H:%M:%S')}] ALERT: Drowsiness detected for {driver_type}!")
                                log_event(f"ALERT: Drowsiness detected for {driver_type}!", level="warning")
                        else:
                            eye_status = "Open"
                            COUNTER = 0
                            if DROWSINESS_ALERT:
                                DROWSINESS_ALERT = False
                                arduino_handler.send('0')
                                print(f"[{time.strftime('%H:%M:%S')}] Alert deactivated.")
                    else:  # Handle case where landmark extraction failed
                        ear = 1.0
                        eye_status = "Error: Landmarks"
                        COUNTER = 0
                        if DROWSINESS_ALERT: 
                            DROWSINESS_ALERT = False
                            arduino_handler.send('0')

            else:  # No face landmarks detected by MediaPipe
                eye_status = "Open (No Landmarks)"
                ear = 1.0
                COUNTER = 0
                if DROWSINESS_ALERT: 
                    DROWSINESS_ALERT = False
                    arduino_handler.send('0')

            # Display Drowsiness Info
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eyes: {eye_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            alert_color = (0, 0, 255) if DROWSINESS_ALERT else (0, 255, 0)
            alert_text = "WAKE UP!" if DROWSINESS_ALERT else "Alert: Off"
            cv2.putText(frame, alert_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            cv2.putText(frame, f"Total Alerts: {TOTAL_ALERTS}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        stage_times['drowsiness_detection'] = time.time() - drowsy_start_time

        # --- Performance Metrics Display ---
        display_start_time = time.time()
        metrics = perf_monitor.get_current_metrics()
        cv2.putText(frame, f"CPU: {metrics['cpu']:.1f}%", (image_width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Memory: {metrics['memory']:.1f}MB", (image_width-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {metrics['fps']:.1f}", (image_width-200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press ESC to exit", (image_width-200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Stage 3: Display Frame ---
        cv2.imshow('Drowsiness Detection & Driver Recognition', frame)
        stage_times['display'] = time.time() - display_start_time
        
        # Record frame completion for FPS calculation
        perf_monitor.record_frame_processed()
        
        # Print detailed timing every 100 frames
        if frame_count % 100 == 0:
            total_time = sum(stage_times.values())
            print(f"\n--- Frame {frame_count} Processing Breakdown ---")
            for stage, t in stage_times.items():
                percentage = (t / total_time) * 100 if total_time > 0 else 0
                print(f"{stage}: {t*1000:.1f}ms ({percentage:.1f}%)")
            print(f"Total frame time: {total_time*1000:.1f}ms")
            summary = perf_monitor.get_summary()
            print(f"Avg CPU: {summary['avg_cpu']:.1f}%, Peak: {summary['peak_cpu']:.1f}%")
            print(f"Avg Memory: {summary['avg_memory']:.1f}MB, Peak: {summary['peak_memory']:.1f}MB")
            print(f"Avg FPS: {summary['avg_fps']:.1f}, Min: {summary['min_fps']:.1f}, Max: {summary['max_fps']:.1f}")
            print("-----------------------------------")

        # --- Stage 4: Exit ---
        # Changed from 'q' to ESC key (27)
        if cv2.waitKey(5) & 0xFF == 27:  # 27 is the ASCII code for ESC key
            print("Exit requested.")
            break

    # --- Cleanup ---
    print("\n--- Performance Summary ---")
    summary = perf_monitor.get_summary()
    log_event("Final Performance Summary")  # Add log entry
    log_performance(summary)  # Log final metrics
    print(f"Total frames processed: {summary['total_frames']}")
    print(f"Average CPU usage: {summary['avg_cpu']:.1f}%")
    print(f"Peak CPU usage: {summary['peak_cpu']:.1f}%")
    print(f"Average memory usage: {summary['avg_memory']:.1f} MB")
    print(f"Peak memory usage: {summary['peak_memory']:.1f} MB")
    print(f"Average FPS: {summary['avg_fps']:.1f}")
    print(f"FPS range: {summary['min_fps']:.1f} - {summary['max_fps']:.1f}")
    
    # Save performance graph
    perf_monitor.save_performance_graph()
    
    # Stop performance monitoring
    perf_monitor.stop()
    
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    if face_mesh: face_mesh.close()
    # Arduino disconnect is handled in the main block's finally clause


# --- Main Execution ---
if __name__ == "__main__":
    print("--- System Start ---")
    log_event("System Start")  # Add log entry
    # Initialize components
    init_database()  # Ensure DB tables exist
    detector_net, embedder_net = load_face_models()
    arduino = ArduinoComm()  # Create Arduino handler instance

    # Keep track of loaded state for cleanup
    models_loaded = detector_net is not None and embedder_net is not None
    arduino_connected_initially = False

    try:
        if models_loaded:
            # Run UI only if models loaded (UI needs models for encoding)
            print("\nLaunching Driver Manager UI...")
            log_event("Launching Driver Manager UI")  # Add log entry
            print("Close the UI window when finished managing drivers to start detection.")
            run_driver_manager_ui(detector_net, embedder_net)  # Pass models to UI
            print("Driver Manager UI closed.")
            log_event("Driver Manager UI closed")  # Add log entry

            # Load encodings AFTER UI is closed
            known_encodings = load_all_driver_encodings()
            if not known_encodings:
                print("Warning: No known driver encodings loaded. Recognition may not work.")
                log_event("Warning: No known driver encodings loaded. Recognition may not work.", level="warning")  # Add log entry

            # Attempt to connect Arduino AFTER UI
            arduino.connect()
            arduino_connected_initially = arduino.is_connected()  # Store status after connect attempt

            # Run main detection loop
            run_drowsiness_detection(arduino, detector_net, embedder_net, known_encodings)

        else:
            print("\nCritical Error: Face models failed to load. Cannot proceed.")
            log_event("Critical Error: Face models failed to load. Cannot proceed.", level="error")  # Add log entry
            print("Please ensure model files are correctly placed in the 'models' directory.")

    except Exception as main_exception:
        print(f"\n--- An unexpected error occurred in the main execution: ---")
        import traceback
        traceback.print_exc()  # Print detailed traceback
        print(f"Error details: {main_exception}")
        log_event(f"Unexpected error in main execution: {main_exception}", level="error")  # Add log entry
        print("-----------------------------------------------------------")

    finally:
        # Ensure Arduino disconnect is attempted regardless of errors
        print("--- System Shutdown ---")
        log_event("System Shutdown")  # Add log entry
        if arduino and arduino.is_connected():
            arduino.disconnect()
        elif arduino_connected_initially:  # Log if initial connect worked but failed later
            print("Arduino was connected but may have encountered an error.")
            log_event("Arduino was connected but may have encountered an error.", level="warning")  # Add log entry
        else:
            print("Arduino was not connected or connection failed.")
            log_event("Arduino was not connected or connection failed.", level="warning")  # Add log entry
        print("Cleanup complete.")
        log_event("Cleanup complete")  # Add log entry
