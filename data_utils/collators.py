import torch
import dgl
from dgl import transforms as T
from copy import deepcopy

    
class Universal_Collator(object):
    def __init__(self, task, args, device):

        self.device = device
        self.task = task.lower()

        if 'bgrl' == self.task or 'grace' == self.task:
            t1 = dgl.transforms.FeatMask(node_feat_names=['feat'], p=args.p_feat_drop)
            t2 = dgl.transforms.DropEdge(args.p_edge_drop)
            if args.make_undirected:
                t3 = dgl.transforms.AddReverse()
                self.transforms = T.Compose([t1, t2, t3])
            else:
                self.transforms = T.Compose([t1, t2])

        if 'graphmae' == self.task:
            t1 = dgl.transforms.DropEdge(args.p_edge_drop)
            if args.make_undirected:
                t2 = dgl.transforms.AddReverse()
                self.transforms = T.Compose([t1, t2])
            else:
                self.transforms = T.Compose([t1])
            self.p_node_mask = args.p_node_mask

        if 'dgi' == self.task:
            if args.make_undirected:
                self.transform = dgl.transforms.AddReverse()
            else:
                self.transform = None

        if 'lp' == self.task or 'vgae' == self.task:
            self.edge_batch_size = args.edge_batch_size
            self.make_undirected = args.make_undirected

    def __call__(self, gs):
        if 'grace' == self.task:
            g1 = deepcopy(gs[0])
            g2 = deepcopy(gs[0])
            g1 = self.transforms(g1)
            x1 = g1.ndata['feat']
            src, dst = g1.edges()
            e1 = torch.stack([src, dst], dim=1)

            g2 = self.transforms(g2)
            x2 = g2.ndata['feat']
            src, dst = g2.edges()
            e2 = torch.stack([src, dst], dim=1)
            return x1, e1, x2, e2

        if 'graphmae' == self.task:
            g = deepcopy(gs[0])
            g = self.transforms(g)
            x = g.ndata['feat']
            e = torch.stack(g.edges(), dim=1)

            node_mask_rate = self.p_node_mask
            n_node_mask = int(node_mask_rate * g.number_of_nodes())
            masked_indices = torch.randperm(g.number_of_nodes())[:n_node_mask]
            node_mask = torch.ones(g.number_of_nodes(), dtype=torch.bool)
            node_mask[masked_indices] = False
            return x, e, node_mask

        if 'dgi' == self.task:
            g = deepcopy(gs[0])
            if self.transform:
                g = self.transform(g)
            x = g.ndata['feat']
            e = torch.stack(g.edges(), dim=1)
            return x, e

        if 'lp' == self.task or 'vgae' == self.task:
            g = deepcopy(gs[0])
            src, dst = g.edges()
            edges = torch.stack((src, dst), dim=1)
            pos_idx = torch.randperm(edges.shape[0])[:self.edge_batch_size]
            pos_edges = edges[pos_idx]
            neg_edges = torch.randint(0, g.number_of_nodes(), pos_edges.size(), dtype=torch.long)
            mask = torch.ones(edges.size(0), dtype=torch.bool)
            mask[pos_idx] = False

            mp_edges = edges[mask]

            if self.make_undirected:
                reverse_edges = mp_edges.flip(1)
                mp_edges = torch.cat([mp_edges, reverse_edges], dim=0)
          
            x = g.ndata['feat']
            return x, mp_edges, pos_edges, neg_edges

