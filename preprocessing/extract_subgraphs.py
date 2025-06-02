
import dgl
import torch


import torch.nn.functional as F
import os.path
import torch.utils.data as Data
import argparse
import os
import numpy as np

if __name__ == '__main__':
    dgl_root = '.'
    g = dgl.load_graphs(os.path.join(dgl_root, 'directed_graph_np.dgl'))[0][0]
    print(f'#Edges: {g.num_edges()}')
    orig_nids = torch.load('./partition_graphs/orig_nids')
    data_dir = './partition_graphs'
    output_dir = './subgraphs'
    graph_name = 'papers100M'
    n_partitions = int(g.num_nodes() / 10000.0)

    part_data = dgl.distributed.load_partition(os.path.join(data_dir, f'{graph_name}.json'), 0)
    _, _, _, partition_book, _, _, _ = part_data
    
    subgraph_list = []
    for i in range(n_partitions):
        nids = orig_nids[partition_book.partid2nids(i)]
        sg = dgl.node_subgraph(g, nids)
        subgraph_list.append(sg)
    
    dgl.save_graphs(os.path.join(output_dir, f'{graph_name}_subgraphs.dgl'), subgraph_list)
    print('Saved.')
