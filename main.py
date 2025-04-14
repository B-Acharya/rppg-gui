from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import queue


class rPPGExtractor(QThread):
    result_ready = pyqtSignal(np.ndarray)  # Send back processing result

    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=1)  # Queue for frame processing
        self.running = True

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
            else:
                cropped_head = cv_img
                print("Invalid crop dimensions, using full image")
        else:
            cropped_head = cv_img
            print("No faces detected in the input image.")

        # Extract green channel average (you might want to improve this for actual rPPG)
        if cropped_head.size > 0:
            green = np.mean(cropped_head[:, :, 1])  # Mean of green channel
        else:
            green = 0
        return np.array([green])


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG GUI Demo")
        self.disply_width = 640
        self.display_height = 480

        # Initialize data for the plot
        self.rppg_data = []
        self.max_data_points = 250  # Maximum number of data points to display

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        # Create the plot widget
        self.rppg_widget = pg.PlotWidget()
        self.rppg_widget.setTitle("rPPG Signal")
        self.rppg_widget.setLabel("left", "Amplitude")
        self.rppg_widget.setLabel("bottom", "Samples")
        self.rppg_curve = self.rppg_widget.plot(pen="g")

        # create a text label
        self.textLabel = QLabel("rPPG-GUI")

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.rppg_widget)

        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        # create the processor thread (only one instance that lives throughout the app)
        self.processor = rPPGExtractor()
        self.processor.result_ready.connect(self.update_rppg)
        self.processor.start()

        # start the video thread
        self.thread.start()

    def closeEvent(self, event):
        # Properly stop all threads
        self.thread.stop()
        self.processor.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_rppg(self, green_value):
        """Updates the rPPG plot with new data"""
        # Append new value
        self.rppg_data.append(float(green_value[0]))

        # Keep only the last max_data_points
        if len(self.rppg_data) > self.max_data_points:
            self.rppg_data = self.rppg_data[-self.max_data_points :]

        # Update the plot
        self.rppg_curve.setData(self.rppg_data)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        # Send the frame to the processor thread
        self.processor.process_frame(cv_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio
        )
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
