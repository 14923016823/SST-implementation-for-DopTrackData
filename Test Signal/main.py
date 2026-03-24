import numpy as np
import matplotlib.pyplot as plt
from Create_Test_Data import Generate_Test_Signal
from Plot_Test_Signal import plot_test_signal
from SST import STFT

if __name__ == "__main__":
    # Generate test signal
    t, x = Generate_Test_Signal(Mode='1')

    # Plot the test signal
    plot_test_signal(t, x)

    # Compute the STFT of the test signal
    f_sampling = 25000
    window_size = 1024
    overlap = 512
    t_stft, f_stft, stft_matrix = STFT(x, f_sampling, window_size, overlap)

    # Plot the magnitude of the STFT
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t_stft, f_stft, np.abs(stft_matrix.T), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('STFT Magnitude')
    plt.colorbar(label='Magnitude')
    plt.ylim(0, f_sampling/2)
    plt.show()