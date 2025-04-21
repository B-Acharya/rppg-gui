import dataclasses
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2
import queue
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class faceExtractionResutls:
    image: Optional[np.ndarray] = None
    face_detected: bool = False
    # Have default face coordinates to begin with
    # TODO: Maybe the center is a better fit here ?
    face_coordinates: Tuple[int, int, int, int] = (0, 0, 0, 0)


class FaceExtraction(QThread):

    extracted_face = pyqtSignal(faceExtractionResutls)  # Send back processing result

    def __init__(self, logger):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=5)  # Queue for frame processing
        self.running = True
        self.logger = logger
        self.face_extraction_results = faceExtractionResutls()

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
                self.face_extraction_results.image = frame
                self.process(frame)
                self.extracted_face.emit(self.face_extraction_results)
            except queue.Empty:
                # No frame available, continue waiting
                continue

    def stop(self):
        """Safely stop the thread"""
        self.running = False
        self.wait()

    def process(self, cv_img):

        # TODO This is just opencv face extractor and can be updated to something else ?
        frame_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        faces = self.face_cascade.detectMultiScale(frame_gray)

        if len(faces) > 0:
            # Assuming the first face is the target
            x, y, w, h = faces[0]

            self.face_extraction_results.face_coordinates = (x, y, w, h)
            self.face_extraction_results.face_detected = True

        else:
            self.face_extraction_results.face_detected = False
