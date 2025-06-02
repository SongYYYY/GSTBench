import gc
import os
import time
import dgl
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset

data_root = './original/ogbn_papers100M/'
process_mode = 'CON'

def tokenize_ogb_paper_datasets(labels, chunk_size=2000000):
    def merge_by_ids(meta_data, node_ids):
        meta_data.columns = ["ID", "Title", "Abstract"]
        meta_data["ID"] = meta_data["ID"].astype(np.int64)
        meta_data.columns = ["ID", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="right", on="ID")  # how='left'
        return data

    def merge_tit_abs(title, abs):
        title.columns = ["ID", "Title"]
        title["ID"] = title["ID"].astype(np.int64)
        abs.columns = ["ID", "Abstract"]
        abs["ID"] = abs["ID"].astype(np.int64)
        data = pd.merge(title, abs, how="outer", on="ID", sort=True)
        data.to_csv(f'{data_root}titleabs.tsv', sep="\t", header=True, index=False)
        import gc
        del data
        gc.collect()
        return

    def read_ids_and_labels():
        category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  #
        print('-------------read nid2pid--------------')
        paper_ids = pd.read_csv(paper_id_path_csv, usecols=[0])
        print('-------------read label2class--------------')
        categories = pd.read_csv(category_path_csv)
        categories.columns = ["ID", "category"]  # 指定ID 和 category列写进去
        paper_ids.columns = ["ID"]
        categories.columns = ["label_id", "category"]
        paper_ids["label_id"] = labels  # labels 与 ID 相对应; ID mag_id, labels
        return categories, paper_ids  # 返回类别和论文ID

    def process_raw_text_df(meta_data, node_ids):
        b = time.time()
        data = merge_by_ids(meta_data, node_ids)
        print(f'waste {time.time() - b} in merge_by_ids')

        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
            'CON': con_filter,
        }
        # Merge title and abstract
        data['text'] = data.apply(text_func[process_mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:512]), axis=1) #d.cut_off
        return data

    def con_filter(x):
        res = ''
        if isinstance(x['title'], str):
            res += x['title']
        res += '.'
        if isinstance(x['abstract'], str):
            res += x['abstract']

        return res 
        

    if not os.path.exists(f'{data_root}titleabs.tsv'):
        print('--------start read abstract--------')
        abstract = pd.read_csv(f'{data_root}paperinfo/idx_abs.tsv', sep='\t', header=None)
        print('abstract ok')
        title = pd.read_csv(f'{data_root}paperinfo/idx_title.tsv', sep='\t', header=None)
        print('title ok')
        print('------------begin merge------------')
        merge_tit_abs(title, abstract)
        print('merge ok')

    print('found existing titleabs.tsv!')
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    s = time.time()
    categories, node_ids = read_ids_and_labels()
    print(f'waste {time.time()-s} in read_ids_and_labels')
    raw_text_path = f'{data_root}titleabs.tsv'
    for i, meta_data in enumerate(tqdm(pd.read_table(raw_text_path, chunksize=chunk_size))):  # chunksize=chunk_size,
        a = time.time()
        processed_chunk = process_raw_text_df(meta_data, node_ids)
        print(f'waste {time.time() - a} in process_raw_text_df')
        # tokenized = tokenizer(text['text'].tolist(), padding='max_length', truncation=True, max_length=512).data
    #     for k in d.token_keys:
    #         token_info[k][text['ID']] = np.array(tokenized[k], dtype=d.info[k].type)
    # uf.pickle_save('processed', d._processed_flag['token'])
        processed_chunk.to_csv('./data/processed_data_con.csv', mode='a', index=False, header=(i == 0))

    return

if __name__ == '__main__':
    print('-------------loading ognb dataset------------')
    data = DglNodePropPredDataset('ogbn-papers100M', root='./original')
    g, labels = data[0]
    labels = labels.squeeze().numpy()
    np.save('labels.npy', labels)
    print('labels saved.')
    print('-------------loading ognb lebels------------')
    labels = np.load('./labels.npy')
    tokenize_ogb_paper_datasets(labels)
    print('-------------processing over----------------')