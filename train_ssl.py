import utils 
import dgl
import torch
from lr import WarmupCosineDecayScheduler, ConstantLRScheduler
import os.path
import argparse
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import init_process_group, estimate_remaining_time, worker_init_fn
from torch.utils.data import DataLoader, DistributedSampler
from data_utils.collators import Universal_Collator
from data import LazyGraphDataset
from pretrain_model.GraphMAE import PretrainGraphMAE
from pretrain_model.VGAE import PretrainVGAE
from pretrain_model.DGI import PretrainDGI
from pretrain_model.GRACE import PretrainGRACE
from pretrain_model.LP import PretrainLP
from pretrain_model.base_models import GATNet, GCNNet
from torch.nn.parallel import DistributedDataParallel
import time 
import datetime
import gc   
import random
from eval_helper import get_node_data_all, eval_downstream, create_k_shot_tasks
from copy import deepcopy
from pathlib import Path

def train(dataloader, sampler, model, optimizer, scheduler, task, 
          epochs, rank, device, ckpt_path, nc_data, nc_tasks, args):
        num_batches = len(dataloader)
        init_time = time.time()

        if rank == 0:
            print(f'------------Initial-Evaluate-------------')
            nc_dict = eval_downstream(model.module, nc_data, nc_tasks, device, args)
            ave_acc_val, ave_acc_test, ave_count = 0, 0, 0
            for data_name in nc_dict.keys():
                for method_name in nc_dict[data_name].keys():
                    print("DATA: {} | METHOD: {} \n" \
                        "VAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}". \
                        format(data_name, method_name, 
                        nc_dict[data_name][method_name][0], nc_dict[data_name][method_name][1],
                        nc_dict[data_name][method_name][2], nc_dict[data_name][method_name][3]))
                    
                    ave_acc_val += nc_dict[data_name][method_name][0]
                    ave_acc_test += nc_dict[data_name][method_name][2]
                    ave_count += 1

            ave_acc_val /= ave_count
            ave_acc_test /= ave_count
            print('--------------------------------')
            print(f'AVE-ALL-VAL: {ave_acc_val:.5f} | AVE-ALL-TEST: {ave_acc_test:.5f}')
            print('--------------------------------')

        # best results
        best_val_acc = 0
        best_test_acc = 0
        best_epoch = 0

        # training
        print(f'------------Training-------------')
        for epoch in range(epochs):
            model.train()
            sampler.set_epoch(epoch)
            epoch_loss_mean = 0
            loss_count = 0
            step = 0
            start_time = time.time()

            for data in dataloader:
                optimizer.zero_grad()
                train_loss = model(data)  
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                train_loss.backward()
                optimizer.step()
                scheduler.step()

                # record loss
                epoch_loss_mean += train_loss.item()
                loss_count += 1

                if rank == 0:
                    step += 1
                    estimate_remaining_time(start_time, step, num_batches, 100)
            
            if rank == 0:
                print('EPOCH {:05d} | TRAIN LOSS: {:.5f}'.format(
                    epoch+1, epoch_loss_mean/loss_count))
                
           # evaluate per epoch
            if rank == 0 and epoch % args.eval_step == 0:
                print(f'------------Evaluate-{epoch+1}------------')
                nc_dict = eval_downstream(model.module, nc_data, nc_tasks, device, args)
                ave_acc_val, ave_acc_test, ave_count = 0, 0, 0
                for data_name in nc_dict.keys():
                    for method_name in nc_dict[data_name].keys():
                        print("DATA: {} | METHOD: {} \n" \
                            "VAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}". \
                            format(data_name, method_name, 
                            nc_dict[data_name][method_name][0], nc_dict[data_name][method_name][1],
                            nc_dict[data_name][method_name][2], nc_dict[data_name][method_name][3]))
                        
                        ave_acc_val += nc_dict[data_name][method_name][0]
                        ave_acc_test += nc_dict[data_name][method_name][2]
                        ave_count += 1
                
                ave_acc_val /= ave_count
                ave_acc_test /= ave_count
                print('--------------------------------')
                print(f'AVE-ALL-VAL: {ave_acc_val:.5f} | AVE-ALL-TEST: {ave_acc_test:.5f}')
                print('--------------------------------')
                if ave_acc_val > best_val_acc:
                    best_val_acc = ave_acc_val
                    best_test_acc = ave_acc_test
                    best_res_dict = deepcopy(nc_dict)
                    best_epoch = epoch+1
                print(f'BEST-VAL-ACC: {best_val_acc:.5f} | BEST-TEST-ACC: {best_test_acc:.5f} | BEST-EPOCH: {best_epoch}')
                print('--------------------------------')
                
            if rank == 0 and epoch % args.save_step == 0:
                torch.save(model.module.state_dict(), os.path.join(ckpt_path, f'{task}-{epoch}.ckpt'))

            if rank == 0:
                epoch_time = time.time()
                running_time = epoch_time - init_time
                formatted_time = str(datetime.timedelta(seconds=running_time))
                print('TOTAL RUNNING TIME: EPOCH-{}: {}.'.format(epoch+1, formatted_time))
                print('--------------------------------')

        if rank == 0:
            print(f'------------Final-Evaluate-------------')
            print(f'-----Best Result from Epoch {best_epoch}-----')
            for data_name in best_res_dict.keys():
                for method_name in best_res_dict[data_name].keys():
                    print("DATA: {} | METHOD: {} \n" \
                        "VAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}". \
                        format(data_name, method_name, 
                        best_res_dict[data_name][method_name][0], best_res_dict[data_name][method_name][1],
                        best_res_dict[data_name][method_name][2], best_res_dict[data_name][method_name][3]))
            print('--------------------------------')
            print(f'BEST-VAL-ACC: {best_val_acc:.5f} | BEST-TEST-ACC: {best_test_acc:.5f}')
            print('--------------------------------')

def get_model(device, args):
    if args.encoder_name == 'GAT':
        hidden_dim = args.hidden_dim // args.n_head
        encoder = GATNet(384, hidden_dim, args.hidden_dim, args.n_layers, feat_drop=args.dropout, attn_drop=args.attn_drop, heads=args.n_head, 
            norm=args.norm, activation=args.activation, use_residual=args.use_residual)
    elif args.encoder_name == 'GCN':
        encoder = GCNNet(384, args.hidden_dim, args.hidden_dim, args.n_layers, dropout=args.dropout,  
            norm=args.norm, activation=args.activation, use_residual=args.use_residual)
    else:
        raise ValueError(f'Not implemented: {args.encoder_name}.')

    if args.task.lower() == 'graphmae':
        pretrain_model = PretrainGraphMAE(encoder, device, args)
    elif args.task.lower() == 'vgae':
        pretrain_model = PretrainVGAE(encoder, device, args)
    elif args.task.lower() == 'dgi':
        pretrain_model = PretrainDGI(encoder, device, args)
    elif args.task.lower() == 'grace':
        pretrain_model = PretrainGRACE(encoder, device, args)
    elif args.task.lower() == 'lp':
        pretrain_model = PretrainLP(encoder, device, args)
    else:
        raise ValueError(f'Not implemented: {args.task}.')

    return pretrain_model 

def load_subgraphs(data_dir, graph_name):
    n_node_list = []
    n_edge_list = []
    subgraphs = dgl.load_graphs(os.path.join(data_dir, f'{graph_name}_subgraphs.dgl'))[0]
    for i in range(len(subgraphs)):
        sg = subgraphs[i]
        n_node_list.append(sg.num_nodes())
        n_edge_list.append(sg.num_edges())

    return subgraphs, n_node_list, n_edge_list

def main(rank, world_size, args):
    if args.port == 12345:
        args.port = random.randint(10000, 65535)
    init_process_group(world_size, rank, args.port)
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
        print(f'Using GPU {device}')
    else:
        device = torch.device("cpu")
    
    utils.set_random_seed(args.seed)

    ckpt_path = f'ckpt-{args.log_id}'
    Path(ckpt_path).mkdir(exist_ok=True)

    # prepare pretraining data
    if 'papers100M' == args.pretrain_data:
        data_dir = '/localscratch/GFM_data/papers100M/subgraphs'
        graph_name = 'papers100M'
        feature_path = '/localscratch/GFM_data/papers100M/emb/sbert_embeddings_con_split.npy'
        graph_list, n_node_list, n_edge_list = load_subgraphs(data_dir, graph_name)
        if args.pretrain_data_ids[0] != -1:
            graph_list = [graph_list[id] for id in args.pretrain_data_ids]
        elif args.pretrain_data_size != -1:
            selected_ids = torch.randperm(len(graph_list))[:args.pretrain_data_size]
            graph_list = [graph_list[id] for id in selected_ids]
        dataset = LazyGraphDataset(graph_list, feature_path)
    else:
        try:
            pyg_graph = torch.load(os.path.join(args.eval_data_dir, f'{args.pretrain_data}_fixed_sbert.pt'))
        except:
            pyg_graph = torch.load(os.path.join(args.eval_data_dir, f'{args.pretrain_data}.pt'))
        src, dst = pyg_graph.edge_index
        graph = dgl.graph((src, dst), num_nodes=pyg_graph.num_nodes)
        graph.ndata['feat'] = pyg_graph.x
        dataset = [graph]

    collator = Universal_Collator(args.task, args, device)

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    dataloader = DataLoader(dataset=dataset, batch_size=1, sampler=sampler,
                    collate_fn=collator, worker_init_fn=worker_init_fn, num_workers=args.num_workers, pin_memory=True)

    model = get_model(device, args).to(device)

    if device.type == "cpu":
        model = DistributedDataParallel(model, find_unused_parameters=False)
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device, find_unused_parameters=False
        )

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.module.trainable_parameters(), lr=0.0, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.module.trainable_parameters(), lr=0.0, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.module.trainable_parameters(), lr=0.0, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        tot_steps = len(dataloader) * args.epochs
        warmup_steps = len(dataloader) if args.warmup_steps == -1 else args.warmup_steps
        print(f'TOTAL OPTIM STEPS: {tot_steps} | WARMUP STEPS: {warmup_steps}.')
        scheduler = WarmupCosineDecayScheduler(
                    optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=tot_steps,
                    max_lr=args.peak_lr,
                )
    elif args.scheduler == 'constant':
        scheduler = ConstantLRScheduler(optimizer, args.peak_lr)
    else:
        raise ValueError(f'Unrecognized scheduler: {args.scheduler}.')

    # downstream data
    nc_data = get_node_data_all(args.eval_data_names, args.eval_data_dir)
    nc_tasks = {}
    for data_name, data in nc_data.items():
        nc_tasks[data_name] = create_k_shot_tasks(args.eval_task_dir, data_name, data['y'], args.n_tasks, args.n_shots, args.n_val, args.eval_data_seed)

    gc.collect()

    utils.set_random_seed(args.seed)
    train(
        dataloader, 
        sampler,
        model,
        optimizer,
        scheduler,
        args.task, 
        args.epochs,
        rank,
        device,
        ckpt_path,
        nc_data,
        nc_tasks,
        args
        )
    
    print("Optimization Finished!")
    print('--')
    dist.destroy_process_group()


if __name__ == '__main__':
    default_downstream_root = os.environ.get('downstream_root', None)
    default_task_root = os.environ.get('task_root', None)
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--pretrain_data', type=str)
    parser.add_argument('--pretrain_data_size', type=int, default=-1)
    parser.add_argument('--eval_data_names', metavar='N', type=str, nargs='+')
    parser.add_argument('--pretrain_data_ids', metavar='N', type=int, nargs='+')
    parser.add_argument('--eval_data_dir', type=str, default=default_downstream_root)
    parser.add_argument('--eval_task_dir', type=str, default=default_task_root)


    # model parameters
    parser.add_argument('--encoder_name', type=str, default='GCN2')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--norm', type=str, default='none')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--use_residual', action='store_true')


    # training parameters
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--make_undirected', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--peak_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eval_step',  type=int, default=1)
    parser.add_argument('--save_step',  type=int, default=1)

    # ssl parameters
    parser.add_argument('--task', type=str)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--edge_batch_size',  type=int, default=5000)
    parser.add_argument('--p_edge_drop', type=float, default=0.2)
    parser.add_argument('--p_feat_drop', type=float, default=0.2)
    parser.add_argument('--p_node_mask', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.5)

    # eval parameters
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--eval_data_seed', type=int, default=0)

    # linear probing parameters
    parser.add_argument('--linear_runs', type=int, default=1)
    parser.add_argument('--linear_lr', type=float, default=1e-2)
    parser.add_argument('--linear_l2', type=float, default=0)
    parser.add_argument('--linear_dropout', type=float, default=0.1)

    # ft parameters
    parser.add_argument('--ft_runs', type=int, default=1)
    parser.add_argument('--ft_lr', type=float, default=1e-3)
    parser.add_argument('--ft_l2', type=float, default=0)
    parser.add_argument('--ft_dropout', type=float, default=0.1)
    parser.add_argument('--ft_max_epochs', type=int, default=100)
    parser.add_argument('--ft_n_trainable_layers', type=int, default=-1)

    #general parameters
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_id', type=str)


    args = parser.parse_args()
    print(args)
    num_gpus = args.num_gpus

    mp.spawn(main, args=(num_gpus, args), nprocs=num_gpus)


