import logging
import sys
import time

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fft_extraction import CalcualteFFT
from rppg_extraction import rPPGExtractor
from face_extraction import faceExtractionResutls, FaceExtraction

# 1. Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        logger.info("Starting the App")
        self.setWindowTitle("rPPG GUI Demo")
        self.disply_width = 640
        self.display_height = 480

        # Initialize data for the plot
        self.rppg_data = []
        self.rppg_timestamps = []
        self.max_data_points = 250  # Maximum number of data points to display
        self.hr = 0.0

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        # Create the plot widget
        self.rppg_widget = pg.PlotWidget()
        self.rppg_widget.setTitle("rPPG Signal")
        self.rppg_widget.setLabel("left", "Amplitude")
        self.rppg_widget.setLabel("bottom", "Samples")
        self.rppg_curve = self.rppg_widget.plot(pen="g")

        self.fft_widget = pg.PlotWidget()
        self.fft_widget.setTitle("FFT of the Signal")
        self.fft_widget.setLabel("left", "Power")
        self.fft_widget.setLabel("bottom", "Frequency")
        self.fft_cruve = self.fft_widget.plot(pen="g")

        # create a text label
        self.textLabel = QLabel("rPPG-GUI")
        self.hr_widget = QLabel(f"Current HR: {self.hr}")

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.rppg_widget)
        vbox.addWidget(self.fft_widget)
        vbox.addWidget(self.hr_widget)

        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.videoThread = VideoThread()
        self.videoThread.change_pixmap_signal.connect(self.start_processing)
        self.videoThread.start()

        # create the face extraction thread
        self.face_extraction = FaceExtraction(logger)
        self.face_extraction.extracted_face.connect(self.update_image)
        self.face_extraction.start()

        # create the processor thread (only one instance that lives throughout the app)
        self.rppg_extractor = rPPGExtractor(logger)
        self.rppg_extractor.result_ready.connect(self.update_rppg)
        self.rppg_extractor.start()

        # create the process thread to calcuate the HR
        self.calculateFFT = CalcualteFFT(logger)
        self.calculateFFT.fft_estimate_singal.connect(self.update_FFT)
        self.calculateFFT.start()

        self.calculateFFT.fft_estimate_singal.connect(self.update_HR)

    def closeEvent(self, event):
        # Properly stop all threads
        self.videoThread.stop()
        self.rppg_extractor.stop()
        self.calculateFFT.stop()
        event.accept()

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_FFT(self, frequency: np.ndarray, fftSgnal: np.ndarray):
        self.fft_cruve.setData(
            x=frequency,
            y=fftSgnal.reshape(-1),
        )

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_HR(self, frequency: np.ndarray, fftSignal: np.ndarray):
        max = np.argmax(fftSignal)
        self.hr = frequency[max] * 60.0
        self.hr_widget.setText(f"Current HR: {self.hr}")

    @pyqtSlot(np.ndarray)
    def update_rppg(self, green_value: np.ndarray):
        """Updates the rPPG plot with new data"""
        # Append new value
        current_time = current_time = time.time()

        self.rppg_data.append(float(green_value[0]))
        self.rppg_timestamps.append(current_time)

        # Apply filtering only if we have enough data
        if len(self.rppg_data) > 30:
            # Keep only the last max_data_points
            fs = self.estimate_fs(self.rppg_timestamps)
            logger.info(f"fs:{fs}")
            filtered = self.calculateFFT.filter_signal(self.rppg_data, fs=fs)
            self.rppg_curve.setData(filtered)

            if len(self.rppg_data) > self.max_data_points:
                self.rppg_data = self.rppg_data[-self.max_data_points :]
                self.calculateFFT.process_signal(filtered, fs)

        else:
            self.rppg_curve.setData(self.rppg_data)

    @pyqtSlot(np.ndarray)
    def start_processing(self, webcam_img: np.ndarray):
        # Send the frame to the processor thread
        self.face_extraction.process_frame(webcam_img)

    @pyqtSlot(faceExtractionResutls)
    def update_image(self, face_extraction_results: faceExtractionResutls):
        """Updates the image_label with a new opencv image"""

        qt_img = self.convert_cv_qt(face_extraction_results)
        self.image_label.setPixmap(qt_img)
        self.rppg_extractor.process_frame(face_extraction_results)

    def convert_cv_qt(self, face_extraction_results):
        """Convert from an opencv image to QPixmap"""

        rgb_image = cv2.cvtColor(face_extraction_results.image, cv2.COLOR_BGR2RGB)

        # overlay a retangular box over it
        x, y, w, h = face_extraction_results.face_coordinates

        if face_extraction_results.face_detected:
            # use green if the face was detected for this frame
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # use red when the face was not detected and an old face extraction is used
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio
        )
        return QPixmap.fromImage(p)

    def estimate_fs(self, timestamps):

        if len(timestamps) < 2:
            return 30  # Default fallback

        intervals = np.diff(timestamps)
        avg_interval = np.mean(intervals)
        fs = 1.0 / avg_interval if avg_interval > 0 else 30  # Avoid divide-by-zero
        return fs


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
