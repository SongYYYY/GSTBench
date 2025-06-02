import torch
import torch.nn.functional as F
from torch import Tensor 
from torch_sparse import SparseTensor
import torch.nn as nn

class PretrainGRACE(torch.nn.Module):
    def __init__(self, encoder, device, args):
        super(PretrainGRACE, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                    nn.ELU(), 
                                    nn.Linear(args.hidden_dim, args.hidden_dim))
        self.act = nn.ReLU()
        self.tau = args.tau
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for module in self.projector:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

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
        x1, e1, x2, e2 = data
        x1, e1, x2, e2 = x1.to(device), e1.to(device), x2.to(device), e2.to(device)
        A1 = SparseTensor.from_edge_index(e1.t().to(device), torch.ones(e1.shape[0]).to(device), [x1.shape[0], x1.shape[0]])
        A2 = SparseTensor.from_edge_index(e2.t().to(device), torch.ones(e2.shape[0]).to(device), [x2.shape[0], x2.shape[0]])

        z1 = self.encoder(x1, A1) 
        z1 = self.act(z1)
        z2 = self.encoder(x2, A2) 
        z2 = self.act(z2)   

        h1 = self.projector(z1)
        h2 = self.projector(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        loss = (l1 + l2) * 0.5
        loss = loss.mean()

        return loss


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())