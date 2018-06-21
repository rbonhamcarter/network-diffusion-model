# Module containing functions for implementation of network diffusion model
# Created May 18, 2018
import numpy as np
from scipy import linalg
from scipy import stats


def symmetric_normalized_laplacian(matrix):
    """Returns the symmetric normalized network Laplacian. Input should be a square matrix"""

    # Compute the degree of each node in the network corresponding to the matrix given
    delta = np.sum(matrix, axis=1)  # row sum = node degree

    # Initialise the vector that will hold the inverse square root of the node degrees
    n = len(delta)  # number of nodes
    delta_inv_sqrt = np.zeros(n)

    # Fill the vector
    if delta.any() == 0:  # Check for unconnected nodes, this prevents a division by zero error
        for i in range(0, n):
            if delta[i] == 0:
                delta_inv_sqrt[i] = 0  # set the inverse square root of the degree of each unconnected node to zero
            else:
                delta_inv_sqrt[i] = 1.0 / (np.sqrt(delta[i]))
    else:
        delta_inv_sqrt = 1.0 / (np.sqrt(delta))  # skip checking and looping through if all elements are non-zero

    # Compute the network Laplacian, L
    diag_inv_sqrt = np.diag(delta_inv_sqrt)  # create diagonal matrix equal to the degree matrix^(-1/2)
    laplacian = np.eye(n) - diag_inv_sqrt.dot(matrix).dot(diag_inv_sqrt)

    return laplacian


def functional_connectivity(laplacian, beta, time):
    """This function returns a sequence of functional connectivity matrices over time, generated via a network
    diffusion process informed by the laplacian of the structural connectivity network and a decay rate beta.
    The model is from "Network diffusion accurately models the relationship between structural and functional brain
    connectivity networks", Abdelnour et al, 2014. The time vector gives the diffusion time"""

    # Initialise the array holding the sequence of functional connectivity matrices
    n = len(laplacian)  # number of brain regions/network nodes
    functional_conn = np.zeros((n, n, len(time)))

    # Fill the array at each time step
    for i in range(0, len(time)):
        functional_conn[:, :, i] = linalg.expm(-1*beta*laplacian*time[i])

    return functional_conn


def find_t_critical(modeled_functional, true_functional, model_time_dimension=None, true_time_dimension=None):
    """This function returns scalars informing on the optimisation of the similarity between the input arrays. The
    measure of similarity used here is the Pearson correlation coefficient between matrices. If the input arrays are
    sequences of matrices over time, the dimension of each the array that specifies time should be input as an integer
    for model_time_dimension and true_time_dimension. The function returns:

    t_critical_modeled and t_critical_true - the time in each original time scale at which the Pearson correlation
                                            coefficient is maximized between the modeled and true functional
                                            connectivity matrices given.
    pearson_max - the maximum Pearson correlation coefficient computed between any modeled and true functional matrix
                    pair
    p_value - the p-value associated with the pearson_max value
    model_index - the time dimension index of the modeled functional matrix in the modeled_functional array that gives
                    pearson_max
    true_index - the time dimension index of the true functional matrix in the true_functional array that gives
                    pearson_max
    """

    # Initialise the vectors
    Pearson_coefficients = []
    p_values = []

    # Check for each time dimension case and then compute the desired statistics
    if model_time_dimension is None and true_time_dimension is None:  # neither have a time dimension
        if len(modeled_functional.shape) != 2 or len(true_functional.shape) != 2:
            raise (ValueError('The input arrays must be 2D!'))
        r = stats.pearsonr(modeled_functional.ravel(), true_functional.ravel())
        Pearson_coefficients.append(r[0])  # correlation coefficient between matrices
        p_values.append(r[1])  # p-value for correlation coefficient between these two matrices
        t_critical_model = None  # no time dimension
        t_critical_true = None  # no time dimension
        pearson_max = max(Pearson_coefficients)
        index = Pearson_coefficients.index(pearson_max)
        p_value = p_values[index]
        model_index = None  # single model matrix
        true_index = None  # single true matrix

    elif model_time_dimension is None:  # no modeled matrix time dimension
        if len(modeled_functional.shape) != 2:
            raise (ValueError('The modeled_functional array must be 2D!'))
        if len(true_functional.shape) != 3:
            raise (ValueError('The true_functional array must be 3D!'))
        if true_time_dimension not in range(3):  # i.e. it's not 0, 1, or 2
            raise (ValueError('The true_time_dimension must be either 0, 1, or 2!'))
        # Determine the length of the true functional time vector
        true_time_length = true_functional.shape[true_time_dimension]
        # Loop through the time steps and compare the single modeled matrix and each true functional matrix
        for i in range(true_time_length):
            # select true functional matrix at time step i depending on which dimension is time
            if true_time_dimension == 0:
                true_functional_i = true_functional[i, :, :]
            elif true_time_dimension == 1:
                true_functional_i = true_functional[:, i, :]
            else:  # true_time_dimension == 2
                true_functional_i = true_functional[:, :, i]
            r = stats.pearsonr(modeled_functional.ravel(), true_functional_i.ravel())
            Pearson_coefficients.append(r[0])
            p_values.append(r[1])
        # Compute the max correlation coefficient and the p-value, times, and indicies associated with it
        pearson_max = max(Pearson_coefficients)
        index = Pearson_coefficients.index(pearson_max)
        p_value = p_values[index]
        model_index = None  # single model matrix
        true_index = index
        t_critical_model = None  # no time dimension
        if true_time_dimension == 0:
            t_critical_true = true_functional[true_index, 0, 0]
        elif true_time_dimension == 1:
            t_critical_true = true_functional[0, true_index, 0]
        else:  # true_time_dimension == 2
            t_critical_true = true_functional[0, 0, true_index]

    elif true_time_dimension is None:  # no true matrix time dimension
        if len(true_functional.shape) != 2:
            raise (ValueError('The true_functional array must be 2D!'))
        if len(modeled_functional.shape) != 3:
            raise (ValueError('The modeled_functional array must be 3D!'))
        if model_time_dimension not in range(3):  # i.e. it's not 0, 1, or 2
            raise (ValueError('The model_time_dimension must be either 0, 1, or 2!'))
        # Determine the length of the model functional time vector
        model_time_length = modeled_functional.shape[model_time_dimension]
        # Loop through the time steps and compare the single true matrix and each modeled functional matrix
        for i in range(model_time_length):
            # select the modeled functional matrix at time step i depending on which dimension is time
            if model_time_dimension == 0:
                modeled_functional_i = modeled_functional[i, :, :]
            elif model_time_dimension == 1:
                modeled_functional_i = modeled_functional[:, i, :]
            else:  # model_time_dimension == 2
                modeled_functional_i = modeled_functional[:, :, i]
            r = stats.pearsonr(modeled_functional_i.ravel(), true_functional.ravel())
            Pearson_coefficients.append(r[0])
            p_values.append(r[1])
        # Compute the max correlation coefficient and the p-value, times, and indicies associated with it
        pearson_max = max(Pearson_coefficients)
        index = Pearson_coefficients.index(pearson_max)
        p_value = p_values[index]
        model_index = index  # single model matrix
        true_index = None
        t_critical_true = None  # no time dimension
        if model_time_dimension == 0:
            t_critical_model = modeled_functional[model_index, 0, 0]
        elif model_time_dimension == 1:
            t_critical_model = modeled_functional[0, model_index, 0]
        else:  # model_time_dimension == 2
            t_critical_model = modeled_functional[0, 0, model_index]

    else:  # both input arrays have a time dimension
        if model_time_dimension not in range(3):  # i.e. it's not 0, 1, or 2
            raise (ValueError('The model_time_dimension must be either 0, 1, or 2!'))
        if true_time_dimension not in range(3):  # i.e. it's not 0, 1, or 2
            raise (ValueError('The true_time_dimension must be either 0, 1, or 2!'))
        # Determine the length of the time vectors
        model_time_length = modeled_functional.shape[model_time_dimension]
        true_time_length = true_functional.shape[true_time_dimension]
        # Loop through the time steps and compare each possible pair of matrices
        # Loop through modeled functional time steps
        for i in range(model_time_length):
            if model_time_dimension == 0:
                modeled_functional_i = modeled_functional[i, :, :]
            elif model_time_dimension == 1:
                modeled_functional_i = modeled_functional[:, i, :]
            else:  # model_time_dimension == 2
                modeled_functional_i = modeled_functional[:, :, i]
            # Loop through true functional time steps
            for j in range(true_time_length):
                if true_time_dimension == 0:
                    true_functional_j = true_functional[j, :, :]
                elif true_time_dimension == 1:
                    true_functional_j = true_functional[:, j, :]
                else:  # true_time_dimension == 2
                    true_functional_j = true_functional[:, :, j]
                r = stats.pearsonr(modeled_functional_i.ravel(), true_functional_j.ravel())
                Pearson_coefficients.append(r[0])
                p_values.append(r[1])
        # Compute the max correlation coefficient and the p-value, times, and indicies associated with it
        pearson_max = max(Pearson_coefficients)
        index = Pearson_coefficients.index(pearson_max)
        p_value = p_values[index]
        # Reshape to determine loop indicies
        loop_shape_Pearson_coefficients = np.reshape(Pearson_coefficients, (model_time_length, true_time_length))
        index_reshape = np.argwhere(loop_shape_Pearson_coefficients == pearson_max)
        model_index = int(index_reshape.ravel()[0])  # if multiple elements in index array, use the first one
        true_index = int(index_reshape.ravel()[1])  # if multiple elements in index array, use the first one
        # Find modeled time at which coefficient is maximised
        if model_time_dimension == 0:
            t_critical_model = modeled_functional[model_index, 0, 0]
        elif model_time_dimension == 1:
            t_critical_model = modeled_functional[0, model_index, 0]
        else:  # model_time_dimension == 2
            t_critical_model = modeled_functional[0, 0, model_index]
        # Find true time at which coefficient is maximised
        if true_time_dimension == 0:
            t_critical_true = true_functional[true_index, 0, 0]
        elif true_time_dimension == 1:
            t_critical_true = true_functional[0, true_index, 0]
        else:  # true_time_dimension == 2
            t_critical_true = true_functional[0, 0, true_index]

    return t_critical_model, t_critical_true,  pearson_max, p_value, model_index, true_index


def distance_between_regions(region_positions):
    """This function returns a symmetric matrix of the Euclidean distance (Frobenius norm) between regions.
    Input is an array of each regions centre position in 3D in some co-ordinate system"""

    # Initialise vector for storing distances
    n = len(region_positions)  # number of regions
    half_distances = np.zeros((n, n))
    # Fill vector
    for i in range(n):
        for j in range(0, i+1):  # only compute half of matrix as it's symmetric
            half_distances[i, j] = np.linalg.norm(region_positions[i, :] - region_positions[j, :])
    # Fill the other side of the matrix
    distances = half_distances + half_distances.T

    return distances


def backward_difference_derivative(signal, time):
    """Returns the discrete time derivative of the signal. The derivative is computed via backward difference over
    each time step"""

    # Initialise vector for storing the derivative
    n = len(time)
    signal_derivative = np.zeros(n)
    # Set the derivative at time step=0 to be equal to the derivative at time step=1
    signal_derivative[0] = (signal[1] - signal[0])/(time[1] - time[0])
    # Compute the derivatives at the rest of the time steps
    for i in range(1, n):
        signal_derivative[i] = (signal[i] - signal[i-1])/(time[i] - time[i-1])

    return signal_derivative



