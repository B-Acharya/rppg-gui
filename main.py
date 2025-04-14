import logging
import queue
import sys

import cv2
import numpy as np
import pyqtgraph as pg
import scipy
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fft_extraction import CalcualteFFT
from rppg_extraction import rPPGExtractor

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

        self.fft_widget = pg.PlotWidget()
        self.fft_widget.setTitle("FFT of the Signal")
        self.fft_widget.setLabel("left", "Power")
        self.fft_widget.setLabel("bottom", "Frequency")
        self.fft_cruve = self.fft_widget.plot(pen="g")

        # create a text label
        self.textLabel = QLabel("rPPG-GUI")

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.rppg_widget)
        vbox.addWidget(self.fft_widget)

        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.videoThread = VideoThread()
        # connect its signal to the update_image slot
        self.videoThread.change_pixmap_signal.connect(self.update_image)
        # start the video thread
        self.videoThread.start()

        # create the processor thread (only one instance that lives throughout the app)
        self.processor = rPPGExtractor(logger)
        self.processor.result_ready.connect(self.update_rppg)
        self.processor.start()

        # create the process thread to calcuate the HR
        self.calculateFFT = CalcualteFFT(logger)
        self.calculateFFT.fft_estimate_singal.connect(self.update_FFT)
        self.calculateFFT.start()

    def closeEvent(self, event):
        # Properly stop all threads
        self.videoThread.stop()
        self.processor.stop()
        self.calculateFFT.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_FFT(self, fftSgnal: np.ndarray):
        self.fft_cruve.setData(fftSgnal.reshape(-1))

    @pyqtSlot(np.ndarray)
    def update_rppg(self, green_value: np.ndarray):
        """Updates the rPPG plot with new data"""
        # Append new value
        self.rppg_data.append(float(green_value[0]))

        # Keep only the last max_data_points
        if len(self.rppg_data) > self.max_data_points:
            self.rppg_data = self.rppg_data[-self.max_data_points :]
            self.calculateFFT.process_signal(self.rppg_data)

        # Update the plot
        self.rppg_curve.setData(self.rppg_data)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img: np.ndarray):
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
    logger = logging.getLogger(__name__)
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
