import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

def convert_to_single_direction(graph):
    """
    Optimized function to convert a DGL graph to a graph with single-direction edges.

    Args:
    graph (dgl.DGLGraph): The DGL graph to be converted.

    Returns:
    dgl.DGLGraph: A new graph with single-direction edges.
    """
    # Extract edges from the graph
    src, dst = graph.edges()

    # Use numpy for efficient processing
    min_edges = np.minimum(src.numpy(), dst.numpy())
    max_edges = np.maximum(src.numpy(), dst.numpy())

    print('TAG-1')
    # Create unique edges
    unique_edges = np.unique(np.vstack((min_edges, max_edges)), axis=1)
    print('TAG-2')
    # Create a new graph with these edges
    new_graph = dgl.graph((unique_edges[0], unique_edges[1]), num_nodes=graph.number_of_nodes())

    return new_graph


if __name__ == '__main__':
    dgl_root = './original'
    data = DglNodePropPredDataset('ogbn-papers100M', root=dgl_root)
    g, labels = data[0]

    print(f'#Edges (Original): {g.num_edges()}')
    new_graph = convert_to_single_direction(g)
    print(f'#Edges (New): {new_graph.num_edges()}')

    print('saving...')
    dgl.save_graphs('./directed_graph_np.dgl', [new_graph])
    print('saved.')

  