import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser, join
import diffusionBBC


def main():
    # Specifying the directory and the files containing the TVB simulated functional connectivity matrices
    home = expanduser('~')
    sim_directory_name = join(home, 'TVB_Distribution', 'Resting_State_Simulated')
    sim_func_conn = np.load(join(sim_directory_name, 'sim_func_conn_0.npy'))  # load in a subject to get array shape

    # Initialise matrix storing all subjects
    n_subject = 11  # number of subjects
    n_time = len(sim_func_conn[0, 0, :])  # number of correlation matrices over time per subject (i.e. # of time steps)
    n_matrix = len(sim_func_conn)  # matrix size
    sim_func_conn_all_subjects = np.zeros((n_subject, n_matrix, n_matrix, n_time))

    # Fill matrix by importing from the file for each subject
    for i in range(n_subject):
        subject = i
        func_filename = 'sim_func_conn_%d.npy' % subject
        sim_func_conn_all_subjects[i, :, :, :] = np.load(join(sim_directory_name, func_filename))

    # Computing the mean over all subjects
    mean_matrix = sim_func_conn_all_subjects.mean(0)

    # Compute max correlation (over time) between individual simulated and group average matrices
    subject = 0  # select subject for comparison
    mean_matrix = np.delete(sim_func_conn_all_subjects, subject, 0).mean(0)  # mean not including subject
    subject_matrix = sim_func_conn_all_subjects[subject, :, :, :]
    (t_critical_model, t_critical_true, pearsonr, p_value, model_index, true_index) = \
        diffusionBBC.find_t_critical(mean_matrix, subject_matrix, model_time_dimension=2, true_time_dimension=2)
    print(t_critical_model, t_critical_true, pearsonr, p_value)


main()
