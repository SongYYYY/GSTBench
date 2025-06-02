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
    data_root = '.'
    chunk_size = 200000
    device = torch.device('cuda:1')
    split_id = 2 
    raw_text_path = os.path.join(data_root, f'processed_data_con_{split_id}.csv')
    # raw_text_path = os.path.join('test_1m.csv')
    tensor_dir = 'tensor_chunks_con'
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    
    for i, meta_data in enumerate(tqdm(pd.read_csv(raw_text_path, chunksize=chunk_size))):  # chunksize=chunk_size,
        a = time.time()
        
        texts = meta_data['text'].tolist()
        ids = meta_data['ID'].tolist()
        
        # Compute Sbert embeddings for the texts in the chunk
        chunk_embeddings = model.encode(texts, show_progress_bar=False, batch_size=2048, device=device)
        # Convert embeddings to tensor and save to disk
        chunk_embeddings = torch.tensor(chunk_embeddings)
        # chunk_embeddings = torch.concat([torch.tensor(ids, dtype=torch.float32).unsqueeze(1), chunk_embeddings], dim=1)
        torch.save(torch.tensor(ids, dtype=torch.long), os.path.join(tensor_dir, f'id_{i}_split_{split_id}.pt'))
        torch.save(chunk_embeddings, os.path.join(tensor_dir, f'chunk_{i}_split_{split_id}.pt'))
        print('id_chunk-{} saved.'.format(i))
        b = time.time()
        print('-----------finish one chunk in {} seconds-------------'.format(b-a))
    
   
    print('---------------------over---------------------')