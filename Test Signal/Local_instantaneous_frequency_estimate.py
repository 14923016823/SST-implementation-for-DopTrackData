import numpy as np

def Local_Instantaneous_Frequency_Estimate(stft_matrix, f, f_sampling, window_size, overlap):
    num_windows, N = stft_matrix.shape
    step = window_size - overlap

    t = np.arange(num_windows) * step / f_sampling
    f = np.fft.fftshift(f)  

    phase = np.angle(stft_matrix)
    phase_prev = np.empty_like(phase)
    phase_prev[0, :] = phase[0, :]
    phase_prev[1:, :] = phase[:-1, :]

    delta_phase = np.angle(np.exp(1j * (phase - phase_prev)))

    dt = step / f_sampling
    instantaneous_frequency_matrix = f[np.newaxis, :] + delta_phase / (2 * np.pi * dt)
    instantaneous_frequency_matrix[np.abs(stft_matrix) == 0] = 0.0

    return t, instantaneous_frequency_matrix



        

