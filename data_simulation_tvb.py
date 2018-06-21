import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser, join
from diffusionBBC import functional_connectivity

# Importing the structural connectivity matrices from TVB default
home = expanduser('~')
directory_name = join(home, 'Documents')
structural_data_file_name = join(directory_name, 'sim_conn_weights.npy')
structural_data_set = np.load(structural_data_file_name)

# Importing the functional connectivity matrices from TVB simulation using the above connectivity
home = expanduser('~')
directory_name = join(home, 'Documents')
func_data_file_name = join(directory_name, 'sim_func_conn.npy')
func_data_set = np.load(func_data_file_name)

# Display both matrices
plt.figure('Simulated Structural and Functional Connectivity Matrices', figsize=(10, 4))
plt.subplot(121)
plt.imshow(structural_data_set, interpolation='none')
plt.subplot(122)
plt.imshow(np.log10(func_data_set)*(func_data_set < 0.05), interpolation='none')
plt.show()

# Compute the weighted degree of each node
delta = np.sum(structural_data_set, axis=1)
n = len(delta)
delta_i_sqrt = 1.0 / (np.sqrt(delta))

# Compute the network Laplacian, L
Delta_i_sqrt = np.diag(delta_i_sqrt)
L = np.eye(n) - Delta_i_sqrt.dot(structural_data_set).dot(Delta_i_sqrt)

# Model the functional connectivity matrix
beta = 2  # decay rate
t = 2  # time
C_f = functional_connectivity(L, beta, t)
plt.figure('Functional Connectivity Matrix, beta = %f, t = %f' % (beta, t))
plt.imshow(C_f)
plt.show()
