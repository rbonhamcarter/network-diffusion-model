import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from os.path import expanduser, join


def main():
    # Import a numpy array as the adjacency matrix
    home = expanduser('~')
    directory_name = join(home, 'Documents', 'data_conn')
    data_file_name = join(directory_name, 'func_modeled.npy')
    struct_conn = np.load(data_file_name)

    # Generate graph
    G = nx.from_numpy_array(struct_conn)

    # Visualize graph
    plt.figure('Graph')
    print(G.number_of_edges())
    nx.draw(G, node_size=50, node_color=np.arange(144),  width=0.1, cmap='Blues')
    plt.show()


main()
