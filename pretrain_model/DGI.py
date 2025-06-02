import dgl
import torch
import torch.nn.functional as F
from torch import Tensor 
from torch_sparse import SparseTensor
import torch.nn as nn

class PretrainDGI(torch.nn.Module):
    def __init__(self, encoder, device, args):
        super(PretrainDGI, self).__init__()
        self.encoder = encoder
        self.discriminator = nn.Bilinear(args.hidden_dim, args.hidden_dim, 1)
        self.act = nn.PReLU(args.hidden_dim)
        self.sigm = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        torch.nn.init.xavier_uniform_(self.discriminator.weight.data)
        if self.discriminator.bias is not None:
            self.discriminator.bias.data.fill_(0.0)
        if isinstance(self.act, nn.PReLU):
            self.act.weight.data.fill_(0.25)  # Default PReLU initialization

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
        # output = self.act(output) # (N, d)

        return output # (N, d)


    def forward(self, data):
        device = self.device
        x, edges = data
        x, edges = x.to(device), edges.to(device)
        A = SparseTensor.from_edge_index(edges.t().to(device), torch.ones(edges.shape[0]).to(device), [x.shape[0], x.shape[0]])
 
        z = self.encoder(x, A) 
        z = self.act(z)
        g = self.sigm(z.mean(dim=0))
        g = g.unsqueeze(0).expand_as(z)
        xn = x[torch.randperm(x.shape[0])]
        zn = self.encoder(xn, A)
        zn = self.act(zn)
        s = self.discriminator(z, g).squeeze()
        sn = self.discriminator(zn, g).squeeze()
        logits = torch.cat([s, sn], dim=0)
        labels = torch.cat([torch.ones(s.shape[0]), torch.zeros(sn.shape[0])], dim=0).to(device)
        loss = self.criterion(logits, labels)

        return loss





