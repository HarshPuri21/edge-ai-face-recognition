# Advanced Edge AI Face Recognition Application

This project is a high-performance, real-time face recognition application designed specifically for deployment on resource-constrained edge devices like the NVIDIA Jetson Nano. It is built with Python, OpenCV, and Dlib, and features a robust set of functionalities that demonstrate an understanding of both classic and modern computer vision pipelines.

---

## Key Features & Technical Highlights

-   **Dual Detection Pipelines:** The application implements two distinct face detection backends, allowing for a real-time trade-off between speed and accuracy:
    -   **HOG + SVM:** A classic, lightweight pipeline that is extremely fast and well-suited for high-FPS applications on edge devices.
    -   **Lightweight CNN:** A more modern, deep learning-based detector that offers higher accuracy, especially for faces at various angles, at the cost of being more computationally intensive.
    -   Users can switch between these two modes with a single key press to observe the performance differences.

-   **Dynamic On-the-Fly Enrollment:** The system is not limited to a pre-defined set of faces. A new person can be enrolled into the recognition database in real-time by capturing their face from the video stream and entering their name via the command line.

-   **Real-time Performance Monitoring:** An on-screen display provides a live **Frames Per Second (FPS)** counter. This is a critical metric for any edge AI application, providing tangible feedback on the system's efficiency and the performance impact of switching between the HOG and CNN detectors.

-   **Optimized for Edge Devices:** The code includes optimizations crucial for edge computing, such as processing resized frames to increase throughput and using efficient, classic libraries like Dlib that are well-suited for devices with limited processing power.

-   **Object-Oriented Design:** The application logic is encapsulated within a clean, object-oriented `FaceRecognizer` class, demonstrating professional software engineering practices for managing complex state and functionality.

---

## Project Structure

-   `face_recognition_app.py`: A single Python script containing the complete application.
    -   Initializes Dlib's face detectors (HOG and CNN) and face recognition models.
    -   Manages the known faces database in memory.
    -   Handles real-time video capture via OpenCV.
    -   Contains the main processing loop for detection, encoding, and comparison.
    -   Renders the UI, including bounding boxes, names, and performance metrics, on the video stream.

---

## How to Run

### Prerequisites
-   A Python environment.
-   A webcam connected to your system.
-   The following libraries installed:
    ```bash
    pip install opencv-python dlib numpy
    ```
-   **Dlib Models:** You will need to download the pre-trained model files from the Dlib website and place them in the same directory as the script:
    -   `shape_predictor_68_face_landmarks.dat`
    -   `dlib_face_recognition_resnet_model_v1.dat`
    -   `mmod_human_face_detector.dat` (for the CNN detector)

### Execution
1.  Run the script from your terminal:
    ```bash
    python face_recognition_app.py
    ```
2.  A window showing your webcam feed will appear.
3.  **Controls:**
    -   `s`: Switch between the fast HOG detector and the accurate CNN detector.
    -   `a`: Add a new face. Point the camera at a new person, press 'a', and then enter their name in the terminal.
    -   `q`: Quit the application.


