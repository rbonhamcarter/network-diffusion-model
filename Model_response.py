import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser, join
import diffusionBBC


def main():
    # Initialise model parameters
    time = np.linspace(0, 10, 100)  # diffusion time
    beta = 1  # decay rate
    n = 11  # matrix size (= number of nodes in the network), should be odd here

    # Generate 'Dirac matrix' to test model response
    dirac_matrix = np.zeros((n, n))
    center_index = int(np.floor(n/2))  # floor not ceiling as indexing starts at 0
    dirac_matrix[center_index, center_index] = 1  # assuming n is odd and thus the matrix "center" is well-defined

    # Model response
    model_func_conn = diffusionBBC.functional_connectivity(dirac_matrix, beta, time)  # dirac in place of Laplacian

    # Visualising model response
    fig, ax = plt.subplots()
    for i in range(len(time)):
        ax.cla()
        ax.imshow(model_func_conn[:, :, i], cmap='jet')
        ax.set_title("diffusion time = %f" % time[i])
        # Note that using time.sleep does *not* work here!
        plt.pause(0.1)

    # Activation signal response

    # Importing the structural connectivity matrices from dMRI data
    home = expanduser('~')
    directory_name = join(home, 'Documents', 'data_conn')
    data_file_name = join(directory_name, 'c_matrix_new_sym.npy')
    struct_conn = np.load(data_file_name)

    # Computing the symmetric normalized network Laplacian
    laplacian = diffusionBBC.symmetric_normalized_laplacian(struct_conn)
    initial_activation_signal = np.zeros(len(struct_conn))
    initial_activation_signal[center_index] = 1  # "dirac vector"

    # Computing the activation signal time series
    matrix_exponential_term = diffusionBBC.functional_connectivity(laplacian, beta, time)
    activation_signal = np.zeros((len(struct_conn), len(time)))
    for i in range(len(time)):
        activation_signal[:, i] = matrix_exponential_term[:, :, i].dot(initial_activation_signal)

    # Plotting activation signal response
    for i in range(n):
        plt.plot(activation_signal[i, :])
    plt.title('Activation signal response across 144 regions')
    plt.xlabel('time', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.show()


main()
