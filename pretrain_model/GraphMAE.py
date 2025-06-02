import dgl
import torch
import torch.nn.functional as F
from torch import Tensor 
from torch_sparse import SparseTensor
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class PretrainGraphMAE(torch.nn.Module):
    def __init__(self, encoder, device, args):
        super(PretrainGraphMAE, self).__init__()
        self.encoder = encoder
        # self.decoder = GCNConv(args.hidden_dim, 384)
        self.decoder = GATConv(args.hidden_dim, 384, heads=1, dropout=0.1)
        self.enc_to_dec = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 384))

        self.alpha = args.alpha
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.enc_to_dec.reset_parameters()
        nn.init.zeros_(self.enc_mask_token)

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
        x, edges, node_mask = data
        x, edges, node_mask = x.to(device), edges.to(device), node_mask.to(device)
        A = SparseTensor.from_edge_index(edges.t().to(device), torch.ones(edges.shape[0]).to(device), [x.shape[0], x.shape[0]])

        x = x.clone()
        x_target = x[~node_mask].clone()
        x[~node_mask] = 0
        x[~node_mask] += self.enc_mask_token

        h = self.encoder(x, A)
        h = self.enc_to_dec(h)
        h[~node_mask] = 0
        x_recon = self.decoder(h, A)[~node_mask]
        
        loss = sce_loss(x_recon, x_target, self.alpha)
       
        return loss

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss




