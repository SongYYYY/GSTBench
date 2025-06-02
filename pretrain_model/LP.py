import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

EPS = 1e-15
MAX_LOGSTD = 10

class PretrainLP(torch.nn.Module):
    def __init__(self, encoder, device, args):
        super(PretrainLP, self).__init__()
        self.encoder = encoder
        self.score_func = SimpleMLP(args.hidden_dim, args.hidden_dim, 1, 3, 0.1)
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.score_func.reset_parameters()

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.parameters())

    @torch.no_grad()
    def inference(self, x, edges):
        device = self.device
        self.eval()

        x, edges = x.to(device), edges.to(device)
        adj = SparseTensor.from_edge_index(edges.t().to(device), torch.ones(edges.shape[0]).to(device), [x.shape[0], x.shape[0]])

        output = self.encoder(x, adj) # (N, d)

        return output # (N, d)


    def forward(self, data):
        device = self.device
        x, edges, pos_edges, neg_edges  = data
        x, edges, pos_edges, neg_edges = x.to(device), edges.to(device), pos_edges.to(device), neg_edges.to(device)
        A = SparseTensor.from_edge_index(edges.t().to(device), torch.ones(edges.shape[0]).to(device), [x.shape[0], x.shape[0]])
 
        h = self.encoder(x, A) 

        pos_u, pos_v = pos_edges[:, 0], pos_edges[:, 1]
        neg_u, neg_v = neg_edges[:, 0], neg_edges[:, 1]
        
        # Node embeddings for positive and negative links
        h_pos_u, h_pos_v = h[pos_u], h[pos_v]
        h_neg_u, h_neg_v = h[neg_u], h[neg_v]
        h_pos = h_pos_u * h_pos_v
        h_neg = h_neg_u * h_neg_v
        h_all = torch.cat([h_pos, h_neg], dim=0)  # (P+N, d)
        logits = self.score_func(h_all).squeeze() # (P+N,)
        target = torch.cat([torch.ones(pos_edges.shape[0]), torch.zeros(neg_edges.shape[0])]).to(self.device) # (P+N,)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        return loss


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lins[-1](x)
        return logits