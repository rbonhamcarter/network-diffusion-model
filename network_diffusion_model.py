import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser, join
import diffusionBBC


def main():
    # Specifying the directories containing the empirical structural and functional connectivity data
    home = expanduser('~')
    struct_directory_name = join(home, 'Documents', 'data_conn')
    sim_func_directory_name = join(home, 'TVB_Distribution', 'Resting_State_Simulated')

    # Importing the structural connectivity matrices
    struct_data_file_name = join(struct_directory_name, 'new_all_subjects.npy')
    struct_conn_all_subjects = np.load(struct_data_file_name)

    # Selecting one matrix: i.e selecting one subject
    subject = 9  # There are 11 subjects (numbered 0-10) in this data set
    struct_conn = struct_conn_all_subjects[subject, :, :]

    # Visualise the structural connectivity matrix for the subject selected
    plt.figure('Structural Connectivity Matrix')
    plt.imshow(struct_conn, cmap='jet')
    plt.show()
    # fig, ax = plt.subplots()
    # cax = ax.imshow(struct_conn, cmap='jet')
    # # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # cbar = fig.colorbar(cax, ticks=[0, c_max])
    # cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    # plt.show()

    # Load in the simulated functional connectivity for the selected subject from TVB
    func_filename = 'sim_func_conn_%d.npy' % subject
    sim_func_conn = np.load(join(sim_func_directory_name, func_filename))
    # sim_func_conn[sim_func_conn < 0] = 0  # set all negative elements to zero (unclear if should do)

    # Compute the symmetric normalized network Laplacian
    L = diffusionBBC.symmetric_normalized_laplacian(struct_conn)

    # Compute the functional connectivity matrix from the network diffusion model
    beta = 10  # decay rate
    time = np.linspace(0, 10, 100)  # diffusion time
    C_f = diffusionBBC.functional_connectivity(L, beta, time)

    # Compute the max Pearson correlation and the p-value, times and indicies associated with it
    (t_critical_model, t_critical_true, pearsonr, p_value, model_index, true_index) = \
        diffusionBBC.find_t_critical(C_f, sim_func_conn, model_time_dimension=2, true_time_dimension=2)
    print(t_critical_model, t_critical_true, pearsonr, p_value)

    # Visualise the modeled functional connectivity matrix at time = t_critical
    plt.figure('Functional Connectivity Matrix, beta = %f, t = %f' % (beta, t_critical_model))
    cf_max = C_f[:, :, model_index].max()
    plt.imshow(C_f[:, :, model_index] / cf_max, cmap='jet')  # Visualise the normalised matrix
    plt.show()

    # Save the modeled functional connectivity matrix at time = t_critical
    np.save('func_modeled.npy', C_f[:, :, model_index])

    # fig, ax = plt.subplots()
    # cax = ax.imshow(C_f[:, :, index], cmap='jet')
    # cbar = fig.colorbar(cax, ticks=[0, c_max])
    # cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
    # plt.show()

    # Visualise the simulated matrix at time = t_critical
    plt.figure('TVB Simulated Functional Connectivity Matrix, t = %f' % (t_critical_true))
    sim_func_conn_max = sim_func_conn[:, :, true_index].max()
    plt.imshow(sim_func_conn[:, :, true_index] / sim_func_conn_max, cmap='jet')  # Visualise the normalized matrix
    plt.show()


main()

