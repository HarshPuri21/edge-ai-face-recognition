
# face_recognition_app.py
# An advanced, real-time face recognition application designed for edge devices
# like the NVIDIA Jetson Nano. It features switchable detection backends (HOG/CNN),
# dynamic user enrollment, and performance monitoring.

import cv2
import dlib
import numpy as np
import os
import time

# --- 1. Configuration ---
# In a real application, you would download these pre-trained model files from the Dlib website.
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
CNN_FACE_DETECTOR_PATH = "mmod_human_face_detector.dat"

class FaceRecognizer:
    def __init__(self):
        """Initializes the face recognizer with models and known faces."""
        self.detector_mode = "HOG"  # Start with the faster HOG detector
        self.known_face_encodings = []
        self.known_face_names = []

        # --- MOCK MODELS FOR DEMONSTRATION ---
        # This section simulates the behavior of Dlib's models to make the script
        # self-contained and runnable without downloading large model files.
        self._mock_dlib_models()
        
        self.face_detector_hog = dlib.get_frontal_face_detector()
        # In a real scenario, you would load the CNN model like this:
        # self.face_detector_cnn = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTOR_PATH)
        self.face_detector_cnn = self._mock_cnn_detector
        
        self.current_detector = self.face_detector_hog
        
        # Pre-load and encode known faces from a directory (simulated)
        self._encode_known_faces()

    def _mock_dlib_models(self):
        """Creates mock functions to simulate Dlib's predictors and recognizers."""
        print("Using simulated Dlib models for demonstration.")
        self.shape_predictor = lambda img, box: "mock_landmarks"
        self.face_recognizer = lambda img, landmarks: np.random.rand(128)
        # The CNN detector returns objects with a 'rect' attribute
        MockCnnRect = type('obj', (object,), {'rect': dlib.rectangle(100, 100, 300, 300)})
        self._mock_cnn_detector = lambda img, upsample: [MockCnnRect()]

    def _encode_known_faces(self):
        """Simulates loading images and encoding faces of known individuals."""
        print("Encoding known faces...")
        # For this simulation, we create a few mock known faces.
        self.known_face_names.extend(["Harsh", "Alex"])
        self.known_face_encodings.extend([np.random.rand(128), np.random.rand(128)])
        print(f"Finished encoding {len(self.known_face_names)} known faces.")

    def switch_detector(self):
        """Switches between HOG and CNN face detectors."""
        if self.detector_mode == "HOG":
            self.detector_mode = "CNN"
            self.current_detector = self.face_detector_cnn
        else:
            self.detector_mode = "HOG"
            self.current_detector = self.face_detector_hog
        print(f"Switched to {self.detector_mode} detector.")

    def enroll_new_face(self, frame):
        """Captures a face from the frame and adds it to the known faces."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_rects = self.face_detector_hog(rgb_frame, 1) # Use HOG for fast enrollment capture

        if not face_rects:
            print("Enrollment failed: No face detected.")
            return

        # Assume the largest face is the one to enroll
        face_rect = max(face_rects, key=lambda rect: rect.width() * rect.height())
        
        landmarks = self.shape_predictor(rgb_frame, face_rect)
        new_encoding = np.array(self.face_recognizer(rgb_frame, landmarks))

        # Prompt for the name in the console
        cv2.destroyAllWindows() # Temporarily close video window for input
        name = input("Enter the name for the new face and press Enter: ")
        if name:
            self.known_face_encodings.append(new_encoding)
            self.known_face_names.append(name)
            print(f"Successfully enrolled new person: {name}")
        else:
            print("Enrollment cancelled: No name provided.")

    def run(self):
        """Starts the real-time video processing loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        print("\nStarting real-time face recognition...")
        prev_frame_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS for performance monitoring
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_rects = self.current_detector(rgb_small_frame, 1)

            for face_rect in face_rects:
                # CNN detector returns an object with a 'rect' attribute
                if self.detector_mode == 'CNN':
                    face_rect = face_rect.rect

                # Compute encoding
                landmarks = self.shape_predictor(rgb_small_frame, face_rect)
                unknown_encoding = np.array(self.face_recognizer(rgb_small_frame, landmarks))

                # Compare with known faces
                distances = np.linalg.norm(self.known_face_encodings - unknown_encoding, axis=1)
                best_match_index = np.argmin(distances)
                name = "Unknown"
                if distances[best_match_index] < 0.6: # Dlib's recommended threshold
                    name = self.known_face_names[best_match_index]

                # Draw bounding box and label
                top, right, bottom, left = (face_rect.top() * 2, face_rect.right() * 2, face_rect.bottom() * 2, face_rect.left() * 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Display UI text
            ui_y = 30
            cv2.putText(frame, fps_text, (10, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Detector: {self.detector_mode}", (10, ui_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Controls: [s]witch detector | [a]dd face | [q]uit", (10, ui_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Edge AI Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.switch_detector()
            elif key == ord('a'):
                self.enroll_new_face(frame)
                # Re-initialize camera if it was closed for input
                cap = cv2.VideoCapture(0)


        cap.release()
        cv2.destroyAllWindows()
        print("Video stream stopped.")

# --- Main Execution ---
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run()
