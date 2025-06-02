import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
import random


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = dgl_to_scipy_sparse(g)
    # A = g.adj()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc



def re_features(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        

    nodes_features = nodes_features.squeeze()


    return nodes_features


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix




def dgl_to_torch_sparse(g):
    # Get the adjacency matrix in COO format
    src, dst = g.edges()

    # Create the edge index in PyTorch COO format
    edge_index = torch.stack([src, dst], dim=0)

    # If the graph has edge weights, use them as values
    if 'w' in g.edata:
        edge_weights = g.edata['w']
    else:
        # If no edge weights, use a tensor of ones
        edge_weights = torch.ones(edge_index.shape[1])

    # Create the sparse adjacency matrix
    num_nodes = g.num_nodes()
    adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weights, (num_nodes, num_nodes))

    return adj_matrix

def construct_features_sparse(adj_matrix, X, K):
    # Initialize X_new with self-features
    X_new = [X]

    # Current features for propagation
    current_features = X

    # Iteratively propagate features
    for _ in range(K):
        # Sparse matrix multiplication for feature propagation
        current_features = torch.sparse.mm(adj_matrix, current_features)
        X_new.append(current_features)

    # Concatenate along a new dimension to form [N, K+1, d]
    X_new = torch.stack(X_new, dim=1)

    return X_new


def dgl_to_scipy_sparse(g):
    # Get the edges of the graph
    src, dst = g.edges()

    # Optionally, handle edge weights if your graph has them
    if 'edge_weight' in g.edata:
        edge_weight = g.edata['edge_weight'].numpy()
    else:
        edge_weight = torch.ones(src.shape[0]).numpy()  # Use 1s if no weights

    # Number of nodes
    num_nodes = g.num_nodes()

    # Create a SciPy sparse matrix in COO format
    adj_matrix = sp.coo_matrix((edge_weight, (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))

    # Convert to CSR format
    adj_matrix_csr = adj_matrix.tocsr()

    return adj_matrix_csr


def concat_dgl_graphs(graph_list):
    # Check if the graph list is empty
    if not graph_list:
        raise ValueError("The graph list is empty")

    # Concatenate edge connections and adjust edge indices
    src_list = []
    dst_list = []
    offset = 0
    for graph in graph_list:
        src, dst = graph.edges()
        src_list.append(src + offset)
        dst_list.append(dst + offset)
        offset += graph.number_of_nodes()

    src_cat = torch.cat(src_list, dim=0)
    dst_cat = torch.cat(dst_list, dim=0)

    # Create the concatenated graph
    concatenated_graph = dgl.graph((src_cat, dst_cat), num_nodes=offset)

    return concatenated_graph

# Example usage:
# graph1, graph2, graph3 = ... # Define or load your DGL graphs
# big_graph = concat_dgl_graphs([graph1, graph2, graph3])


import torch.distributed as dist
def print_mean_loss(rank, loss, world_size):
    """
    Gather losses from all processes, compute the mean, and print on rank 0.
    
    Args:
    - rank (int): The rank of the current process.
    - loss (torch.Tensor): The loss of the current process.
    - world_size (int): Total number of processes.
    """
    # Ensure that loss is a tensor
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss, device='cuda')

    # Gather all losses to rank 0
    all_losses = [torch.tensor(0.0, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_losses, loss)

    if rank == 0:
        # Only rank 0 computes the mean and prints it
        mean_loss = sum(all_losses) / world_size
        print(f"Mean Loss: {mean_loss.item()}")


def init_process_group(world_size, rank, port=12345):
    dist.init_process_group(
        backend="nccl",  # change to 'nccl' for multiple GPUs
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )

def set_random_seed(seed):
    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python
    random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

import time 
def estimate_remaining_time(start_time, current_batch, total_batches, k):
    """
    Estimates and prints the remaining training time every K batches.

    :param start_time: The time when the epoch started.
    :param current_batch: The current batch number.
    :param total_batches: The total number of batches in the epoch.
    :param k: The function will print the estimated time every K batches.
    """

    if current_batch % k == 0 and current_batch > 0:
        elapsed_time = time.time() - start_time
        batches_processed = current_batch
        avg_time_per_batch = elapsed_time / batches_processed
        remaining_batches = total_batches - current_batch
        estimated_time = avg_time_per_batch * remaining_batches

        # Convert estimated time to minutes and seconds for better readability
        estimated_minutes = int(estimated_time // 60)
        estimated_seconds = int(estimated_time % 60)

        print(f"Estimated Time Remaining: {estimated_minutes}m {estimated_seconds}s")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import os

def ensure_path_exists(path):
    """
    Check if the given path exists. If not, create it.
    Works with both absolute and relative paths.
    
    :param path: str, the path to check/create
    :return: str, the absolute path that now exists
    """
    # Convert to absolute path if it's a relative path
    abs_path = os.path.abspath(path)
    
    # Check if the path exists
    if not os.path.exists(abs_path):
        # If it doesn't exist, create it
        os.makedirs(abs_path, exist_ok=True)
        print(f"Created directory: {abs_path}")
    else:
        print(f"Directory already exists: {abs_path}")
    
    return abs_path