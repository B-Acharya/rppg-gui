from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
import queue
import scipy
from numba import jit


class CalcualteFFT(QThread):

    fft_estimate_singal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, logger):
        super().__init__()
        self.fft_queue = queue.Queue(maxsize=5)  # Queue for frame processing
        self.running = True
        self.logger = logger

    def process_signal(self, wave, fs):
        """Add a frame to the processing queue"""
        # Replace old frame with new one if queue is full
        if self.fft_queue.full():
            try:
                self.fft_queue.get_nowait()
            except queue.Empty:
                pass
        self.fft_queue.put((wave, fs))

    def run(self):
        while self.running:
            try:
                # Get frame with timeout to avoid blocking indefinitely
                frame, fs = self.fft_queue.get(timeout=1.0)
                freqeuncy, power = self.process(frame, fs)
                self.fft_estimate_singal.emit(freqeuncy, power)
                self.logger.info("Loaded the signal for processing")
            except queue.Empty:
                # No frame available, continue waiting
                continue

    def stop(self):
        """Safely stop the thread"""
        self.running = False
        self.wait()

    def process(self, wave, fs):
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT).
        source: rPPG-Toolbox https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/post_process.py
        """

        self.logger.info("Start of fft processing")
        ppg_signal = np.expand_dims(wave, 0)
        N = self._next_power_of_2(ppg_signal.shape[1])

        f_ppg, pxx_ppg = scipy.signal.periodogram(
            ppg_signal, fs=fs, nfft=N, detrend=False
        )
        return self.mask(f_ppg, pxx_ppg)

    @staticmethod
    @jit(nopython=True)
    def mask(f_ppg, pxx_ppg):
        fmask_ppg = np.argwhere((f_ppg >= 0.6) & (f_ppg <= 3.3))
        mask_ppg = np.take(f_ppg, fmask_ppg)

        mask = (f_ppg >= 0.6) & (f_ppg <= 3.3)

        filtered_freqs = f_ppg[mask]
        filtered_power = pxx_ppg[0][mask]  # pxx_ppg shape: (1, N), so index 0

        # Combine for plotting
        fft_signal = np.zeros_like(f_ppg)
        fft_signal[mask] = filtered_power

        return filtered_freqs, filtered_power

    @staticmethod
    def _next_power_of_2(x):
        """Calculate the nearest power of 2."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    @staticmethod
    def filter_signal(data, lowcut=0.7, highcut=2.5, fs=30, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype="band")
        filtered = scipy.signal.filtfilt(b, a, data)
        return filtered
