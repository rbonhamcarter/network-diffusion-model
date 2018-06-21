import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from nilearn import plotting
from os.path import expanduser, join


def main():
    # Import a numpy array as the adjacency matrix
    home = expanduser('~')
    directory_name = join(home, 'Documents', 'data_conn')
    data_file_name = join(directory_name, 'func_modeled.npy')
    struct_conn = np.load(data_file_name)

    # # Generate graph
    # G = nx.from_numpy_array(struct_conn)
    #
    # # Visualize graph - no meaning to position or colour
    # plt.figure('Graph')
    # n_node = G.number_of_nodes()  # number of nodes
    # nx.draw(G, node_size=50, node_color=np.arange(n_node),  width=0.1, cmap='Blues')
    # plt.show()

    # Visualise graph with positions specified by brain region coordinates
    # # Import node positions from file of (x,y,z) co-ordinates from brain region centers
    positions_file_name = join(directory_name, 'region_positions.npy')
    region_positions = np.load(positions_file_name)
    plotting.plot_connectome(struct_conn,
                             region_positions - region_positions.mean(0),
                             title="Structural Connectivity",
                             display_mode='x',
                             alpha=0,
                             node_color='k',
                             node_size=10,
                             edge_cmap='jet',
                             colorbar=True)
    plotting.show()


main()
