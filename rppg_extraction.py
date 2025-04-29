from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2
import queue
from face_extraction import faceExtractionResutls
from numba import jit


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

    def process_frame(self, face_extraction_resutls):
        """Add a frame to the processing queue"""
        # Replace old frame with new one if queue is full
        #
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(face_extraction_resutls)

    def run(self):
        while self.running:
            try:
                # Get frame with timeout to avoid blocking indefinitely
                face_extraction_resutls = self.frame_queue.get(timeout=1.0)
                result = self.process(face_extraction_resutls)
                self.result_ready.emit(result)
            except queue.Empty:
                # No frame available, continue waiting
                continue

    def stop(self):
        """Safely stop the thread"""
        self.running = False
        self.wait()

    def process(self, face_extraction_resutls):

        x, y, w, h = face_extraction_resutls.face_coordinates

        # Calculate the new coordinates and dimensions for a 1:1 aspect ratio

        # Ensure we don't go out of bounds
        if face_extraction_resutls.image is None:
            # dummy green value for the start when the image is of none type ?
            # This should be removed ?
            green = 0.0
        else:

            green = extract_green(face_extraction_resutls.image, x, y, w, h)

        return np.array([green])


# moved to a seperate fuctions for JIT to be allowed, lets make it go brrrrr
# this actually does a lot of things, should be just fuction to extract green signal !! face cropping shoudl be another fucntions on its own


@jit(nopython=True)
def extract_green(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.float32:
    center_x = x + w // 2
    center_y = y + h // 2
    size = int(max(w, h) * 1.0)
    x_new = max(0, center_x - size // 2)
    y_new = max(0, center_y - size // 2)

    height, width = image.shape[:2]
    x_new = min(x_new, width - size)
    y_new = min(y_new, height - size)

    # If size is too large, adjust it
    if x_new + size > width:
        size = width - x_new
    if y_new + size > height:
        size = height - y_new

        # Only crop if dimensions are valid
    if size > 0 and x_new >= 0 and y_new >= 0:
        cropped_head = image[y_new : y_new + size, x_new : x_new + size]
    else:
        cropped_head = image

        # Extract green channel average (you might want to improve this for actual rPPG)
    green = np.mean(cropped_head[:, :, 1])  # Mean of green channel
    return green
