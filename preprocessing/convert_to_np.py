import torch
import numpy as np 
import time 
import os 

if __name__ == '__main__':
    emb_root = './emb'
    start = time.time()
    features = torch.load(os.path.join(emb_root, 'sbert_embeddings_con_split.pt'), map_location='cpu')
    end = time.time()
    print('LOAD TIME INTERVAL: {}'.format(end-start))

    print('Converting to numpy...')
    start = time.time()
    numpy_array = features.numpy()
    end = time.time()
    print('CONVERTING TIME INTERVAL: {}'.format(end-start))

    print('Saving...')
    np.save(os.path.join(emb_root, 'sbert_embeddings_con_split.npy'), numpy_array)
    print('Saved.')