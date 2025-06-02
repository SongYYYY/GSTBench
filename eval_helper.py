import torch
import torch.nn.functional as F
import os 
from copy import deepcopy
import numpy as np
import dgl 
import torch
import pickle
from collections import defaultdict
from utils import set_random_seed

def get_node_data_all(data_names, data_dir):
    node_data_all = defaultdict(dict)
    for data_name in data_names:
        try:
            data = torch.load(os.path.join(data_dir, f'{data_name}_fixed_sbert.pt'))
        except:
            try:
                data = torch.load(os.path.join(data_dir, f'{data_name}.pt'))
            except:
                raise ValueError(f'file does not exist: {data_name}')

        node_data_all[data_name]['x'] = data.x 
        node_data_all[data_name]['y'] = data.y 
        node_data_all[data_name]['edge_index'] = data.edge_index 

    return node_data_all

def save_tasks(tasks, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(tasks, f)

def load_tasks(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_k_shot_tasks(data_root, data_name, labels, M, K, n_val, seed=0):
    """
    Create or load M K-shot tasks.

    Args:
        data_root (str): Directory where the tasks should be saved/loaded from.
        data_name (str): A name to identify the dataset, used for saving/loading tasks.
        labels (torch.Tensor or array-like): The label for each sample in the dataset.
        M (int): Number of tasks.
        K (int): Number of examples per class in the training set (K-shot).
        n_val (int): Total number of examples (across all classes) in the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        list of dicts: Each dict contains three boolean masks (torch.BoolTensor):
            {
                'train_mask': torch.BoolTensor,
                'val_mask':   torch.BoolTensor,
                'test_mask':  torch.BoolTensor
            }
            The masks have the same length as `labels`.
    """
    # If labels is not already a torch.Tensor, convert it:
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    # Construct the filename for saving/loading
    file_name = f"{data_name}_{K}_{n_val}_{M}_{seed}.pkl"
    file_path = os.path.join(data_root, file_name)

    # If a task file already exists, load and return it
    if os.path.exists(file_path):
        return load_tasks(file_path)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Unique labels and number of samples
    unique_labels = torch.unique(labels)
    num_samples = len(labels)  # total number of samples in the dataset

    tasks = []

    # Create M tasks
    for _ in range(M):
        # Create boolean masks (torch) for train, val, and test
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask   = torch.zeros(num_samples, dtype=torch.bool)
        test_mask  = torch.zeros(num_samples, dtype=torch.bool)

        # For each class, pick which samples go to train
        for c in unique_labels:
            # Indices for this class
            class_indices = torch.where(labels == c)[0]

            # Shuffle
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]

            # Number of samples we have in this class
            n_class_samples = len(shuffled_indices)

            # Train indices (K per class)
            n_train = min(K, n_class_samples)
            train_indices = shuffled_indices[:n_train]

            train_mask[train_indices] = True

        # The remaining indices are potential val/test candidates
        remaining_indices = torch.nonzero(~train_mask, as_tuple=True)[0]
        shuffled_remaining = remaining_indices[torch.randperm(len(remaining_indices))]

        # Pick n_val from the remaining as validation
        n_val_actual = min(n_val, len(shuffled_remaining))
        val_indices  = shuffled_remaining[:n_val_actual]
        test_indices = shuffled_remaining[n_val_actual:]

        val_mask[val_indices]   = True
        test_mask[test_indices] = True

        task_dict = {
            'train_mask': train_mask,
            'val_mask':   val_mask,
            'test_mask':  test_mask
        }
        tasks.append(task_dict)

    # Save the tasks to disk for future reuse
    with open(file_path, 'wb') as f:
        pickle.dump(tasks, f)

    return tasks
    
# Embedding eval: ICL
def get_zero_shot_acc(node_features, node_labels, label_embs, test_mask):
    # 1. Compute Prototype Vectors
    prototypes = F.normalize(label_embs, p=2, dim=1)
    # 2. Compute Similarities and Classify Test Nodes
    test_features = node_features[test_mask]
    test_features = F.normalize(test_features, p=2, dim=1)
    similarity = torch.mm(test_features, prototypes.t())
    # similarity = F.cosine_similarity(test_features, prototypes, dim=1)
    predicted_labels = torch.argmax(similarity, dim=1)

    # 3. Compute Accuracy
    test_labels = node_labels[test_mask]
    accuracy = (predicted_labels == test_labels).sum().item() / test_labels.size(0)

    return accuracy

def add_nodes_and_edges(g, proto_features, labels, train_mask):
    g = g.clone()
    num_prototypes = proto_features.shape[0]
    feature_size = proto_features.shape[1]
    num_existing_nodes = g.num_nodes()
    
    # # Add prototype nodes to the graph
    # g.add_nodes(num_prototypes)
    
    # Get all training nodes
    train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze()

    # Get labels for training nodes
    train_labels = labels[train_indices]

    # Compute prototype node indices
    proto_node_indices = num_existing_nodes + train_labels
    
    # Source and destination nodes for adding bidirectional edges
    src_nodes = torch.cat([train_indices, proto_node_indices])
    dst_nodes = torch.cat([proto_node_indices, train_indices])

    # Add bidirectional edges
    g.add_edges(src_nodes, dst_nodes)

    # Assuming the original features are in g.ndata['feat']
    # original_features = g.ndata['feat']
    # combined_features = torch.cat([original_features, proto_features], dim=0)
    # g.ndata['feat'] = combined_features
    g.ndata['feat'][num_existing_nodes:] = proto_features

    return g

def get_mean_prototpye(node_features, node_labels, train_mask):
    # 1. Compute Prototype Vectors
    unique_labels = torch.unique(node_labels[train_mask])
    prototypes = torch.stack([node_features[train_mask][node_labels[train_mask] == label].mean(dim=0) for label in unique_labels])

    return prototypes

def get_icl_results(pretrain_model, data, train_mask, valid_mask, test_mask, device=None):
    x = data['x']
    y = data['y']
    label_embs = torch.zeros((len(torch.unique(y)), x.shape[1]))
    edge_index = data['edge_index']
    src = edge_index[0]
    dst = edge_index[1]
    g = dgl.graph((src, dst), num_nodes=x.shape[0])
    g.ndata['feat'] = x
    g_new = add_nodes_and_edges(g, label_embs, y, train_mask)
    src, dst = g_new.edges()
    edges_new = torch.stack([src, dst], dim=1)
    output = pretrain_model.inference(g_new.ndata['feat'].to(device),edges_new.to(device))
    label_embs_new = output[g.num_nodes():]
    node_embs = output[:g.num_nodes()]

    valid_acc = get_zero_shot_acc(node_embs, y.to(device), label_embs_new, valid_mask.to(device))
    test_acc = get_zero_shot_acc(node_embs, y.to(device), label_embs_new, test_mask.to(device))
    
    return valid_acc, test_acc

def get_mean_icl_results(pretrain_model, data, tasks, args, device=None):
    val_acc_list = []
    test_acc_list = []
    for task in tasks:
        train_mask, val_mask, test_mask = task['train_mask'], task['val_mask'], task['test_mask']
        val_acc, test_acc = get_icl_results(pretrain_model, data, train_mask, val_mask, test_mask, device)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    
    return np.mean(val_acc_list), np.std(val_acc_list), np.mean(test_acc_list), np.std(test_acc_list)

# Embedding Eval: Linear probing
def get_linear_results(node_embs, data, train_mask, val_mask, test_mask,
                    lr, l2, dropout, args, device):
    x = data['x']
    labels = data['y']
    edge_index = data['edge_index']
    n_classes = labels.max().item() + 1
    
    val_acc_list, test_acc_list = [], []
    for i in range(args.linear_runs):
        set_random_seed(i)
        clf = torch.nn.Linear(node_embs.shape[1], n_classes).to(device)

        val_acc, test_acc = train_clf(clf, node_embs, labels, 
                                        train_mask, val_mask, test_mask, 
                                        lr, l2, dropout,
                                        device)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    return np.mean(val_acc_list), np.mean(test_acc_list)

def train_clf(clf, node_embs, labels, train_mask, val_mask, test_mask, lr=0.01, l2=0, dropout=0.2, device=None):
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=l2)
    best_acc = 0
    for e in range(300):
        clf.train()
        optimizer.zero_grad()
        x = F.dropout(node_embs[train_mask].to(device), p=dropout, training=True)
        out = clf(x)
        loss = F.cross_entropy(out, labels[train_mask].to(device))
        loss.backward()
        optimizer.step()
        val_acc, val_loss = evaluate_clf(clf, node_embs, labels, val_mask, device)
        if val_acc > best_acc:
            best_acc = val_acc
            weights = deepcopy(clf.state_dict())
    
    clf.load_state_dict(weights)
    test_acc, test_loss = evaluate_clf(clf, node_embs, labels, test_mask, device)
        
    return best_acc, test_acc

@torch.no_grad()
def evaluate_clf(clf, node_embs, labels, mask, device):
    clf.eval()
    out = clf(node_embs[mask].to(device))
    pred = out.argmax(dim=1)
    correct = pred.eq(labels[mask].to(device)).sum().item()
    acc = correct / mask.sum().item()
    loss = F.cross_entropy(out, labels[mask].to(device)).item()
    return acc, loss

def get_mean_linear_results(pretrain_model, data, tasks, args, device=None):
    val_acc_list = []
    test_acc_list = []
    lr, l2, dropout = args.linear_lr, args.linear_l2, args.linear_dropout

    x = data['x']
    edge_index = data['edge_index']
    node_embs = pretrain_model.inference(x.to(device), edge_index.t().to(device))
    for task in tasks:
        train_mask, val_mask, test_mask = task['train_mask'], task['val_mask'], task['test_mask']
        val_acc, test_acc = get_linear_results(node_embs, data, train_mask, val_mask, test_mask, 
                                        lr, l2, dropout, args, device)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    
    return np.mean(val_acc_list), np.std(val_acc_list), np.mean(test_acc_list), np.std(test_acc_list)


# Parameter Eval: Fine-tuning
def get_ft_results(pretrain_model, data, train_mask, val_mask, test_mask,
                    args, device):
    x = data['x']
    y = data['y']
    edge_index = data['edge_index']
    n_classes = y.max().item() + 1
    lr, l2, dropout = args.ft_lr, args.ft_l2, args.ft_dropout

    val_acc_list, test_acc_list, best_epoch_list = [], [], []
    for i in range(args.ft_runs):
        set_random_seed(i)
        clf = ClassificationModel(deepcopy(pretrain_model.encoder), args.hidden_dim, n_classes, dropout, device).to(device)
        clf.reset_parameters()
        val_acc, test_acc, best_epoch = train_ft(clf, data, 
                                        train_mask, val_mask, test_mask, 
                                        lr, l2, args.ft_max_epochs, args.ft_n_trainable_layers,
                                        device)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        best_epoch_list.append(best_epoch)

    return np.mean(val_acc_list), np.mean(test_acc_list), np.mean(best_epoch_list)

def train_ft(clf, data, train_mask, val_mask, test_mask, lr, l2, epochs, n_trainable_layers, device=None, return_log=False):
    optimizer = torch.optim.AdamW(clf.get_trainable_parameters(n_trainable_layers), 
                    lr=lr, weight_decay=l2)
    best_val = 0
    best_test = 0
    best_epoch = -1
    x = data['x'].to(device)
    labels = data['y'].to(device)
    edge_index = data['edge_index'].to(device)
    loss_list = []
    val_list = []
    for e in range(epochs):
        clf.train()
        optimizer.zero_grad()
        logits = clf(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask].to(device))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        # eval
        clf.eval()
        with torch.no_grad():
            logits = clf(x, edge_index)
        val_acc = compute_acc(logits.argmax(dim=1)[val_mask], labels[val_mask])
        test_acc = compute_acc(logits.argmax(dim=1)[test_mask], labels[test_mask])
        val_list.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc 
            best_epoch = e
    
    if not return_log:
        return best_val, best_test, best_epoch
    else:
        return best_val, best_test, best_epoch, loss_list, val_list

def compute_acc(pred, labels):
    correct = pred.eq(labels.to(pred.device)).sum().item()
    acc = correct / len(labels)

    return acc 

def get_mean_ft_results(pretrain_model, data, tasks, args, device=None):
    val_acc_list = []
    test_acc_list = []
    best_epoch_list = []

    for task in tasks:
        train_mask, val_mask, test_mask = task['train_mask'], task['val_mask'], task['test_mask']
        val_acc, test_acc, best_epoch = get_ft_results(pretrain_model, data, train_mask, val_mask, test_mask, 
                                        args, device)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        best_epoch_list.append(best_epoch)
    
    return np.mean(val_acc_list), np.std(val_acc_list), np.mean(test_acc_list), np.std(test_acc_list), \
        np.mean(best_epoch_list), np.std(best_epoch_list)


class ClassificationModel(torch.nn.Module):
    def __init__(self, encoder, emb_dim, n_classes, dropout, device):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.clf = torch.nn.Linear(emb_dim, n_classes).to(device)

    def reset_parameters(self):
        self.clf.reset_parameters()

    def get_trainable_parameters(self, n_trainable_layers):
        trainable_parameters = self.encoder.get_trainable_parameters(n_trainable_layers) + list(self.clf.parameters())
        return trainable_parameters

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.clf(h)
        return h


def eval_downstream(pretrain_model, all_data, all_tasks, device, args):
    res_dict = defaultdict(dict)
    for data_name, data in all_data.items():
        val_acc_mean, val_acc_std, test_acc_mean, test_acc_std = \
            get_mean_icl_results(pretrain_model, data, all_tasks[data_name], args, device)
        res_dict[data_name]['ICL'] = [val_acc_mean, val_acc_std, test_acc_mean, test_acc_std, 0, 0]
        val_acc_mean, val_acc_std, test_acc_mean, test_acc_std = \
            get_mean_linear_results(pretrain_model, data, all_tasks[data_name], args, device)
        res_dict[data_name]['LINEAR'] = [val_acc_mean, val_acc_std, test_acc_mean, test_acc_std, 0, 0]
        val_acc_mean, val_acc_std, test_acc_mean, test_acc_std, epoch_mean, epoch_std = \
            get_mean_ft_results(pretrain_model, data, all_tasks[data_name], args, device)
        res_dict[data_name]['FT'] = [val_acc_mean, val_acc_std, test_acc_mean, test_acc_std, epoch_mean, epoch_std]

    return res_dict
