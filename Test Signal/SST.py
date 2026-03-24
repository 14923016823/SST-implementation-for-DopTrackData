import numpy as np

def STFT(x, f_sampling, window_size, overlap, window_type=np.hanning):
    # Calculate the dimensions of the STFT
    N = len(x)
    step = window_size - overlap
    num_windows = (N - window_size) // step + 1
    
    # Create the window function
    window = window_type(window_size)
    
    # Calculate time vector and frequency vector
    t = np.arange(num_windows) * step / f_sampling
    f = np.fft.fftfreq(window_size, d = 1/f_sampling)
    
    # Initialize the STFT matrix
    stft_matrix = np.zeros((num_windows, window_size), dtype=np.complex64)
    
    # Compute the STFT
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        stft_matrix[i, :] = np.fft.fftshift(np.fft.fft(x[start:end]*window))
        
    return t, f, stft_matrix

