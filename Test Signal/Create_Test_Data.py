import numpy as np
import matplotlib.pyplot as plt

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
    elif Mode=='2':
        #base math (same as Mode 1)
        base_phase = phi_1[0] + phi_1[1] * (t - t_0) + phi_1[2] * (t - t_0)**2
        base_amplitude = A_1[0] + A_1[1] * (t - t_0) + A_1[2] * (t - t_0)**2
        
        #Set up the phase jumps 
        baud_rate = 500 # The satellite sends # of random phase jumps per second
        samples_per_symbol = fs // baud_rate
        num_symbols = int(np.ceil(len(t) / samples_per_symbol))
        
        #Randomly pick 0 or np.pi for each "block" of time
        random_bits = np.random.choice([0, np.pi], size=num_symbols)
        
        #Stretch those blocks out to match the length of our time array 't'
        phase_jumps = np.repeat(random_bits, samples_per_symbol)[:len(t)]
        
        #Add the random pi jumps directly to the phase 
        total_phase = base_phase + phase_jumps
        
        return t, base_amplitude * np.exp(1j * total_phase)
    else:
        print("Invalid Mode. Please choose '0' or '1'.")
        return False
    
print("1. Generating Fake Satellite Signal...")
# We use Mode '2' to test the random pi phase jumps
t, x = Generate_Test_Signal(Mode='2')


plt.figure(figsize=(10, 4))
plt.plot(t[:2000], np.real(x[:2000]))
plt.title("Close-up of the Raw Satellite Wave (Mode 2)")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage Amplitude")
plt.grid(True)
plt.show()