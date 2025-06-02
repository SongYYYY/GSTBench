# GSTBench: A Benchmark Study on the Transferability of Graph Self-Supervised Learning

This repository contains the implementation of our benchmark study on graph self-supervised learning (SSL) methods. The codebase supports pretraining on large-scale graph datasets and evaluating the transferability of pretrained models across various downstream tasks.

## Basic Information
- **Paper**: [GSTBench: A Benchmark Study on the Transferability of Graph Self-Supervised Learning](link_to_paper)
- **Conference**: CIKM 2024
- **Authors**: Anonymous

## Prerequisites
Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── data_utils/        # Data processing
├── datasets/         # Dataset storage
├── pretrain_model/   # SSL methods (GraphMAE, VGAE, DGI, GRACE, LP)
├── tasks/           # Downstream task storage
├── train_ssl.py     # Main training
├── eval_helper.py   # Evaluation
├── utils.py         # Utilities
├── data.py          # Data loading
├── lr.py            # Learning rate scheduler
└── template_run.sh  # Template running script
```

## Implemented SSL Methods

### 1. Graph Masked Autoencoder (GraphMAE)
- **Paper**: [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803)
- **Key Idea**: Reconstructs masked node features using graph structure
- **Implementation**: `pretrain_model/GraphMAE.py`

### 2. Variational Graph Autoencoder (VGAE)
- **Paper**: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)
- **Key Idea**: Learns latent representations through variational inference
- **Implementation**: `pretrain_model/VGAE.py`

### 3. Deep Graph Infomax (DGI)
- **Paper**: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341)
- **Key Idea**: Maximizes mutual information between local and global representations
- **Implementation**: `pretrain_model/DGI.py`

### 4. Graph Contrastive Learning (GRACE)
- **Paper**: [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131)
- **Key Idea**: Contrastive learning with graph augmentations
- **Implementation**: `pretrain_model/GRACE.py`

### 5. Link Prediction (LP)
- **Paper**: [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691)
- **Key Idea**: Predicts missing edges in the graph
- **Implementation**: `pretrain_model/LP.py`

## Adaptation Methods

### 1. In-Context Learning
- Constructs a prompt graph to integrate labeled nodes
- Leverages the pretrained model's message passing capabilities
- Adapts to new tasks without parameter updates

### 2. Fine-tuning
- Updates all or selected layers of the pretrained model
- Enables dataset-specific adaptation
- Balances between transferability and task performance

### 3. Linear Probing
- Trains a linear classifier on frozen representations
- Evaluates the quality of learned representations
- Measures representation transferability

## Usage

1. Run experiments using the provided script:
```bash
bash template_run.sh
```

## Key Parameters

### Data Parameters
| Parameter | Description | Options |
|-----------|-------------|---------|
| `pretrain_data` | Dataset for pretraining | 'papers100M' or other datasets |
| `eval_data_names` | Datasets for evaluation | One or more dataset names |

### SSL Method Parameters
| Method | Parameters | Description |
|--------|------------|-------------|
| GraphMAE | `alpha` | Masking ratio |
| | `p_node_mask` | Node masking probability |
| VGAE/LP | `edge_batch_size` | Batch size for edge operations |
| GRACE | `p_edge_drop` | Edge dropout probability |
| | `p_feat_drop` | Feature dropout probability |
| | `tau` | Temperature parameter |

### Model Parameters
| Parameter | Description | Options |
|-----------|-------------|---------|
| `encoder_name` | GNN architecture | 'GCN' or 'GAT' |
| `norm` | Normalization type | 'none', 'batch', 'layer' |
| `activation` | Activation function | 'relu', 'gelu', 'prelu' |
| `use_residual` | Use residual connections | True/False |

### Training Parameters
| Parameter | Description | Options |
|-----------|-------------|---------|
| `opt` | Optimizer | 'adam', 'adamw', 'sgd' |
| `scheduler` | Learning rate schedule | 'constant', 'cosine' |
| `warmup_steps` | Warmup steps | Integer |
| `peak_lr` | Peak learning rate | Float |
| `weight_decay` | Weight decay | Float |
| `epochs` | Training epochs | Integer |

### Evaluation Parameters
| Parameter | Description | Options |
|-----------|-------------|---------|
| `n_tasks` | Number of few-shot tasks | Integer |
| `n_shots` | Shots per class | Integer |
| `n_val` | Validation nodes | Integer |
| `eval_data_seed` | Random seed | Integer |

## Pretraining on ogbn-papers100M

Due to the large size of the processed file (~200GB), we do not include the generated files in this repository. Instead, we provide detailed instructions to generate the necessary files locally.

The preprocessing of the ogbn-papers100M dataset consists of two main parts:
1. Processing the graph structure
2. Processing the node features

### Graph Structure Processing

Follow these steps to process the graph structure:

#### 1. Convert the original graph to a unidirectional graph
```bash
python convert_graph_np.py
```
This script converts the original ogbn-papers100M dataset into a unidirectional graph format.

#### 2. Partition the graph
```bash
python partition.py
```
This step partitions the graph into subgraphs using the METIS algorithm.

#### 3. Extract and save subgraphs
```bash
python extract_subgraphs.py
```
This final step extracts the partitioned subgraphs and saves them to disk as `./papers100M_subgraphs.dgl`.

### Node Feature Processing

Follow these steps to process the node features:

#### 1. Process the original text files
```bash
python process.py
```
This script processes the original text files into a structured CSV format.

#### 2. Split the CSV file
```bash
python split_csv.py
```
This step splits the CSV file into multiple parts for parallel processing.

#### 3. Generate embeddings (Part 1)
```bash
python generate_emb.py
```
This script generates textual embeddings using SentenceBERT for the first batch of data.

#### 4. Generate embeddings (Part 2)
```bash
python generate_emb_1.py
```
This script generates textual embeddings for the second batch of data.

#### 5. Merge embeddings
```bash
python merge_embs.py
```
This step merges the separately generated embeddings into a single file.

#### 6. Convert to NumPy format
```bash
python convert_to_np.py
```
The final step converts the merged embeddings into a NumPy array and saves it as `sbert_embeddings_con_split.npy`.

### Output Files

After completing all the preprocessing steps, you should have the following files:

- `papers100M_subgraphs.dgl`: The partitioned graph structure
- `sbert_embeddings_con_split.npy`: The node feature embeddings

### Notes

- The full preprocessing pipeline may take a significant amount of time and computational resources due to the large size of the dataset.
- Make sure you have enough disk space (~200GB) to store the processed files.
- You may need to adjust the parameters in the scripts according to your hardware capabilities.