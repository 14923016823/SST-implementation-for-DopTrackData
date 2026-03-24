import matplotlib.pyplot as plt
from Create_Test_Data import Generate_Test_Signal

def plot_test_signal(t, x):
    ### Create 3D plot of the signal
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(t, x.real, x.imag, zdir='z', label='Complex Test Signal')

    # Make legend, set axis limits and labels
    ax.legend()
    ax.set_xlim(0, 1.03)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Real Part')
    ax.set_zlabel('Imaginary Part')

    ax.view_init(elev=20., azim=-35, roll=0) # Adjust the view angle for better visualization

    plt.show()