import numpy as np

def Generate_Test_Signal(Mode = '0'):
### Define moc signal parameters

    # Signal phase parameters 
    phi_0 = [0, 2 * np.pi * 10, 0] # 2nd order Taylor expansions of phase around t = 0s 
    phi_1 = [0, 2 * np.pi * 1000, 2 * np.pi * 100] # 2nd order Taylor expansions of phase around t = 0s 

    # Signal amplitude parameters
    A_0 = [10, 0, 0] # 2nd order Taylor expansions of amplitude around t = 0s
    A_1 = [10, 2, 0.5] # 2nd order Taylor expansions of amplitude around t = 0s

    # Signal time parameters
    t_0 = 0 # Time of signal start in seconds
    T = 1.03 # Signal duration in seconds

    # Simulated sampled frequency
    fs = 25000 # Sampling frequency in Hz

    # Create moc signla in np.complex64
    t = np.arange(t_0, t_0 + T, 1/fs)
    if Mode == '0':
        return  t, (A_0[0] + A_0[1] * (t - t_0) + A_0[2] * (t - t_0)**2) * np.exp(1j * (phi_0[0] + phi_0[1] * (t - t_0) + phi_0[2] * (t - t_0)**2))
    elif Mode == '1':
        return  t, (A_1[0] + A_1[1] * (t - t_0) + A_1[2] * (t - t_0)**2) * np.exp(1j * (phi_1[0] + phi_1[1] * (t - t_0) + phi_1[2] * (t - t_0)**2))
    else:
        print("Invalid Mode. Please choose '0' or '1'.")
        return False