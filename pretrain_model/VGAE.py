import torch
from torch import Tensor 
from torch_sparse import SparseTensor
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple

EPS = 1e-15
MAX_LOGSTD = 10

class PretrainVGAE(torch.nn.Module):
    def __init__(self, encoder, device, args):
        super(PretrainVGAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() 
        self.vae_output = nn.Linear(args.hidden_dim, 2*args.hidden_dim)
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.vae_output.reset_parameters()

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


    def forward(self, data, return_recon_loss=False):
        device = self.device
        x, edges, pos_edges, neg_edges  = data
        x, edges, pos_edges, neg_edges = x.to(device), edges.to(device), pos_edges.to(device), neg_edges.to(device)
        A = SparseTensor.from_edge_index(edges.t().to(device), torch.ones(edges.shape[0]).to(device), [x.shape[0], x.shape[0]])
 
        z = self.encoder(x, A) 
        output = self.vae_output(z) # (N, 2*d)
        mu, logstd = torch.chunk(output, 2, dim=-1) # (N, d)
        logstd = logstd.clamp(max=MAX_LOGSTD)
        # Reparameterization trick
        z = self.reparametrize(mu, logstd)  # (N, d)
        recon_loss = self.recon_loss(z, pos_edges.t(), neg_edges.t())
        kl_loss = (1 / x.shape[0]) * self.kl_loss(mu, logstd)

        if not return_recon_loss:
            loss = recon_loss + kl_loss
        else:
            return recon_loss

        return loss

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


