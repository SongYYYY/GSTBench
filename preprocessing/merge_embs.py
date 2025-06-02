import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import to_undirected
from transformers import AutoModel, AutoTokenizer
import time
import sys
import os
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer




if __name__ == '__main__':
    data_root = './data'
    chunk_size = 200000
    # raw_text_path = os.path.join('test_1m.csv')
    tensor_dir = 'tensor_chunks_con'
    
    # AFTER ALL CHUNKS 
    sum_embeddings = torch.zeros(384)
    count_embeddings = 0

    # Load each tensor chunk, sum embeddings and count non-missing rows
    for file_name in os.listdir(tensor_dir):
        if file_name.startswith('chunk_') and file_name.endswith('.pt'):
            embeddings_tensor = torch.load(os.path.join(tensor_dir, file_name))
            # embeddings_tensor = embeddings_tensor[:, 1:]
            sum_embeddings += torch.mean(embeddings_tensor, dim=0)
            count_embeddings += 1

    # Calculate the average embeddings for non-missing rows
    average_embedding = sum_embeddings / count_embeddings

    # Initialize the final tensor with NaNs
    final_tensor = torch.full((111059956, 384), float('nan'))
    # final_tensor = torch.full((1010000, 384), float('nan'))

    
    # Populate the final tensor with the saved embeddings and fill missing ones with the average
    for file_name in os.listdir(tensor_dir):
        if file_name.startswith('chunk_') and file_name.endswith('.pt'):
            chunk_id = int(file_name.split('.')[0].split('_')[1]) 
            split_id = int(file_name.split('.')[0].split('_')[3]) 
            ids = torch.load(os.path.join(tensor_dir, f'id_{chunk_id}_split_{split_id}.pt'))
            embeddings_tensor = torch.load(os.path.join(tensor_dir, file_name))
            final_tensor[ids] = embeddings_tensor

    # Find missing rows (where all elements are NaN) and fill them with the average embedding
    missing_indices = torch.isnan(final_tensor[:, 0])
    final_tensor[missing_indices] = average_embedding
    print(f'#TOTAL MISSING: {missing_indices.sum()}.')

    # Save the final tensor to disk
    torch.save(final_tensor, './emb/sbert_embeddings_con_split.pt')
    print('---------------------Save over---------------------')