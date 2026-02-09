import numpy as np

class EEGProcessor:
    def __init__(self, sampling_rate=1000, buffer_size=1024):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros(buffer_size)

    def get_latest_data(self):
        # Generate synthetic EEG signal (alpha waves + noise)
        t = np.linspace(0, self.buffer_size/self.sampling_rate, self.buffer_size)
        alpha = np.sin(2 * np.pi * 10 * t)
        noise = 0.5 * np.random.randn(self.buffer_size)
        return alpha + noise
