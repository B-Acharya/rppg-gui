from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import queue
import scipy


class CalcualteFFT(QThread):

    fft_estimate_singal = pyqtSignal(np.ndarray)

    def __init__(self, logger):
        super().__init__()
        self.fft_queue = queue.Queue(maxsize=5)  # Queue for frame processing
        self.running = True
        self.logger = logger

    def process_signal(self, wave):
        """Add a frame to the processing queue"""
        # Replace old frame with new one if queue is full
        if self.fft_queue.full():
            try:
                self.fft_queue.get_nowait()
            except queue.Empty:
                pass
        self.fft_queue.put(wave)

    def run(self):
        while self.running:
            try:
                # Get frame with timeout to avoid blocking indefinitely
                frame = self.fft_queue.get(timeout=1.0)
                result = self.process(frame)
                self.fft_estimate_singal.emit(result)
                self.logger.info("Loaded the signal for processing")
            except queue.Empty:
                # No frame available, continue waiting
                continue

    def stop(self):
        """Safely stop the thread"""
        self.running = False
        self.wait()

    def process(self, wave):
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT).
        source: rPPG-Toolbox https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/post_process.py
        """

        self.logger.info("Start of fft processing")
        ppg_signal = np.expand_dims(wave, 0)
        N = self._next_power_of_2(ppg_signal.shape[1])

        f_ppg, pxx_ppg = scipy.signal.periodogram(
            ppg_signal, fs=30, nfft=N, detrend=False
        )

        fmask_ppg = np.argwhere((f_ppg >= 0.6) & (f_ppg <= 3.3))
        mask_ppg = np.take(f_ppg, fmask_ppg)

        return mask_ppg

    @staticmethod
    def _next_power_of_2(x):
        """Calculate the nearest power of 2."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
