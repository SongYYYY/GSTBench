import argparse
import time
import os
import dgl
import torch as th


if __name__ == "__main__":
    dgl_root = '.'
    g = dgl.load_graphs(os.path.join(dgl_root, 'directed_graph_np.dgl'))[0][0]
    print(f'#Edges: {g.num_edges()}')

    data_name = 'papers100M'
    num_parts = int(g.num_nodes() / 10000.0)
    output = 'partition_graphs'
    start = time.time()
    orig_nids, orig_eids = dgl.distributed.partition_graph(
        g,
        data_name,
        num_parts,
        output,
        part_method='metis',
        balance_edges=True,
        num_trainers_per_machine=1,
        return_mapping=True,
        num_hops=1,
    )
    end = time.time()
    print('Done partitioning in {}s.'.format(end-start))
    th.save(orig_nids, f'{output}/orig_nids')
    th.save(orig_eids, f'{output}/orig_eids')
    print('Saved.')
    
