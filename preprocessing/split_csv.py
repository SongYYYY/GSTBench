import gc
import os
import time
import dgl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


if __name__ == '__main__':
    data_root = './data'
    raw_text_path = os.path.join(data_root, 'processed_data_con.csv')
    df = pd.read_csv(raw_text_path)
    print('read over.')
    # Define the split point
    split_point = len(df) // 2  # This will split the DataFrame into two nearly equal halves

    # Split the DataFrame
    df1 = df.iloc[:split_point]
    df1.to_csv('processed_data_con_1.csv', index=False)
    print('df1 saved.')
    # del df1 
    # gc.collect()

    df2 = df.iloc[split_point:]
    df2.to_csv('processed_data_con_2.csv', index=False)
    print('df2 saved.')

