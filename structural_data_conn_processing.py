import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser, join

# The script is for processing a set of structural connectivity matrices,
# the region_labels, region_positions, and region_normals have already been processed
# and saved, thus we do not re-process them here


def main():
    # Importing the structural connectivity matrices from dMRI data (~Etienne)
    home = expanduser('~')
    directory_name = join(home, 'Documents', 'data_conn')
    data_file_name = join(directory_name, 'c_matrix_l10.npy')
    # titles_file_name = join(directory_name, 'c_matrix_titles.npy')
    # labels_file_name = join(directory_name, 'c_matrix_data_labels.npy')
    struct_data_set = np.load(data_file_name)
    # Importing region information for processing matrices
    lh_positions = np.load(join(directory_name, 'lh_positions.npy'))
    rh_positions = np.load(join(directory_name, 'rh_positions.npy'))

    # Generate mask to remove regions with undefined characteristics
    region_positions = np.concatenate((lh_positions, rh_positions))
    mask = ~np.isnan(region_positions)
    mask = mask.ravel()
    mask = mask[::3]  # region positions is given in x,y,z co-ordinates so reduce via slicing

    # Selecting subject matrices: index = (subject, acquistion, tractography method used (see labels), rows, columns)
    struct_conn_all_subjects = struct_data_set[:, 0, 7, :, :]  # all subjects, acquistion #1, method = S11A3_100_1_prob
    # Process each matrix
    n_subjects = len(struct_conn_all_subjects[:, 0, 0])  # number of subjects
    n_regions = len(struct_conn_all_subjects[0, :, :])  # number of brain regions
    # Initialise processed data array, -10 as known a priori that is how many regions will be removed
    new_all_subjects = np.zeros((n_subjects, n_regions - 10, n_regions - 10))
    for i in range(n_subjects):
        subject = struct_conn_all_subjects[i, :, :]  # select one subject's connectivity matrix
        new_subject = subject[:-2, :-2]  # Removing data from non-grey matter regions
        new_subject = new_subject[mask == True]  # Removing data from regions with NaN characteristics
        new_subject = new_subject[:, mask == True]
        np.fill_diagonal(new_subject, 0.0)  # Assuming regions are not connected to themselves
        mask2 = ~np.all(new_subject == 0, axis=1)  # Generate mask to remove unconnected regions
        new_subject = new_subject[mask2 == True]  # removing all-zero rows
        new_subject = new_subject[:, mask2 == True]  # removing all-zero columns
        new_subject_sym = (new_subject + new_subject.T)/2.0  # symmetrise the matrix
        new_all_subjects[i, :, :] = new_subject_sym  # store the processed matrix of the subject

    # Save the resulting processed data set
    np.save(join(directory_name, 'new_all_subjects.npy'), new_all_subjects)


main()
