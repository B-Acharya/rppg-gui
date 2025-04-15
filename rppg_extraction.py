from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2
import queue


class rPPGExtractor(QThread):
    result_ready = pyqtSignal(np.ndarray)  # Send back processing result

    def __init__(self, logger):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=5)  # Queue for frame processing
        self.running = True
        self.logger = logger

        self.face_cascade = cv2.CascadeClassifier(
            "/Users/bhargavacharya/PycharmProjects/rppg-gui/haarcascade_frontalface_alt.xml"
        )

    def process_frame(self, frame):
        """Add a frame to the processing queue"""
        # Replace old frame with new one if queue is full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def run(self):
        while self.running:
            try:
                # Get frame with timeout to avoid blocking indefinitely
                frame = self.frame_queue.get(timeout=1.0)
                result = self.process(frame)
                self.result_ready.emit(result)
            except queue.Empty:
                # No frame available, continue waiting
                continue

    def stop(self):
        """Safely stop the thread"""
        self.running = False
        self.wait()

    def process(self, cv_img):
        frame_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        faces = self.face_cascade.detectMultiScale(frame_gray)

        if len(faces) > 0:
            # Assuming the first face is the target
            x, y, w, h = faces[0]

            # Calculate the new coordinates and dimensions for a 1:1 aspect ratio
            center_x = x + w // 2
            center_y = y + h // 2
            size = int(max(w, h) * 1.0)
            x_new = max(0, center_x - size // 2)
            y_new = max(0, center_y - size // 2)

            # Ensure we don't go out of bounds
            height, width = cv_img.shape[:2]
            x_new = min(x_new, width - size)
            y_new = min(y_new, height - size)

            # If size is too large, adjust it
            if x_new + size > width:
                size = width - x_new
            if y_new + size > height:
                size = height - y_new

            # Only crop if dimensions are valid
            if size > 0 and x_new >= 0 and y_new >= 0:
                cropped_head = cv_img[y_new : y_new + size, x_new : x_new + size]
                self.logger.info("Valid face detected full image")
            else:
                cropped_head = cv_img
                self.logger.info("Invalid crop dimensions, using full image")
        else:
            cropped_head = cv_img
            self.logger.info("No faces detected in the input image.")

        # Extract green channel average (you might want to improve this for actual rPPG)
        if cropped_head.size > 0:
            green = np.mean(cropped_head[:, :, 1])  # Mean of green channel
        else:
            green = 0
        return np.array([green])

    def filter_signal(self, data, lowcut=0.7, highcut=4.0, fs=30, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        filtered = filtfilt(b, a, data)
        return filtered
