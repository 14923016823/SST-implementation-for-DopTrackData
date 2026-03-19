import numpy as np

def hilbert_transform(fft_signal):
    """
    Compute the Hilbert transform from an FFT vector.

    Parameters:
    fft_signal (numpy array): FFT of the input signal.

    Returns:
    numpy array: The Hilbert transform of the input signal.
    """

    # Use the FFT 
    N = len(fft_signal)

    # Create a filter to zero out negative frequencies
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    # Apply the filter to the FFT of the signal
    filtered_fft_signal = fft_signal * h

    # Compute the inverse FFT to get the Hilbert transform
    hilbert_signal = np.fft.ifft(filtered_fft_signal)

    return hilbert_signal