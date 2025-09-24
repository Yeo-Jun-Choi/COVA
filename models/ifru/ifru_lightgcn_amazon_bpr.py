import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from utility.load_data import * 
import scipy.sparse as sp
import torch.nn.functional as F
from torch.autograd import Variable
from Model.Lightgcn import LightGCN
from utility.compute import *
import time
import random
import types


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'Amazon_Book'
        self.attack = '0.01'
        self.k_hop = 0
        self.data_type = 'full'
        self.if_epoch = 5000
        self.if_lr = 1e4
        self.if_init_std = 0
        self.seed = 1024
        self.lr = 1e-3
        self.regs = 0
        self.init_std = 1e-3

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0
        self.n_neg = 1
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)


class influence_unlearn(nn.Module):
    def __init__(self,save_name,if_epoch=100,if_lr=1e-2,k_hop=1,init_range=1e-4) -> None:
        super(influence_unlearn).__init__()
        self.if_epoch = if_epoch
        self.if_lr = if_lr
        self.k_hop = k_hop
        self.range = init_range
        self.save_name = save_name
        self.p = None

    def compute_hessian_with_test(self, model=None, data_generator=None, original_model=None, baseline_valid_recall_20=None):
        #    (k-hop=0   )
        nei_users, nei_items = compute_neighbor(data_generator, 0)
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()

        mask = get_eval_mask(data_generator)
        
        #      (k-hop=0  )
        nei_users, nei_items = self.compute_neighbor_influence_clip(data_generator, k_hop=self.k_hop)
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()

        #    
        un_u_para = model.embedding_user.weight[nei_users].reshape(-1)
        un_i_para = model.embedding_item.weight[nei_items].reshape(-1)
        u_para_num = un_u_para.shape[0]
        i_para_num = un_i_para.shape[0]

        un_ui_para = torch.cat([un_u_para,un_i_para],dim=-1)
        u_para = model.embedding_user.weight.clone().detach()
        i_para = model.embedding_item.weight.clone().detach()
        u_para[nei_users] = un_ui_para[:u_para_num].reshape(-1, u_para.shape[-1])
        i_para[nei_items] = un_ui_para[u_para_num:].reshape(-1, i_para.shape[-1])

        def loss_fun(para1, para2):
            u_para,i_para = para1,para2
            learned_data = data_generator.train_original[['user','item','label']].values
            learned_data = torch.from_numpy(learned_data).cuda()
            adj_graph = model.Graph
            user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, adj_graph)
            user_embs = user_emb_gnn[learned_data[:,0].long()]
            item_embs = item_emb_gnn[learned_data[:,1].long()]
            scores = torch.mul(user_embs, item_embs).sum(dim=-1)
            bce_loss = F.binary_cross_entropy_with_logits(scores, learned_data[:,-1].float(), reduction='mean')
            return bce_loss
        
        def unlearn_loss_fun(para1,para2):
            u_para, i_para = para1, para2
            rm_data = data_generator.train_unlearn_original.values
            rm_data = torch.from_numpy(rm_data).cuda()
            adj_graph = model.Graph
            user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, adj_graph)
            user_embs_rm = user_emb_gnn[rm_data[:,0].long()]
            item_embs_rm = item_emb_gnn[rm_data[:,1].long()]
            scores = torch.mul(user_embs_rm, item_embs_rm).sum(dim=-1)
            rm_loss = F.binary_cross_entropy_with_logits(scores, rm_data[:,2].float(), reduction='sum')
            
            # changed data  (/  )
            ch_np = data_generator.changed_data[['user','item','label']].values
            if ch_np.size == 0:
                ch_loss = torch.tensor(0.0, device='cuda')
            else:
                ch_np = ch_np.astype(np.float32, copy=False)
                ch_data = torch.from_numpy(ch_np).cuda()
                user_emb_ch1 = user_emb_gnn[ch_data[:,0].long()]
                item_emb_ch1 = item_emb_gnn[ch_data[:,1].long()]
                ch_score1 = torch.mul(user_emb_ch1,item_emb_ch1).sum(dim=-1)
                ch_loss_1 = F.binary_cross_entropy_with_logits(ch_score1, ch_data[:,2].float(), reduction='sum')
                
                ch_graph = data_generator.ChangedGraph
                user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, ch_graph)
                user_emb_ch2 = user_emb_gnn[ch_data[:,0].long()]
                item_emb_ch2 = item_emb_gnn[ch_data[:,1].long()]
                ch_score_2 = torch.mul(user_emb_ch2,item_emb_ch2).sum(dim=-1)
                ch_loss_2 = F.binary_cross_entropy_with_logits(ch_score_2, ch_data[:,2].float(), reduction='sum')
                
                ch_loss = ch_loss_1 - ch_loss_2
            unlearn_loss = rm_loss + ch_loss
            return unlearn_loss

        total_loss = loss_fun(u_para, i_para)
        total_grad = torch.autograd.grad(total_loss, un_ui_para, create_graph=True, retain_graph=True)[0].reshape(-1,1)
        unlearn_loss = unlearn_loss_fun(u_para, i_para)
        unlearn_grad = torch.autograd.grad(unlearn_loss, un_ui_para,retain_graph=True)[0].reshape(-1,1)

        def hvp(grad, vec):
            vec = vec.detach()
            prod = torch.mul(vec, grad).sum()
            res = torch.autograd.grad(prod, un_ui_para, retain_graph=True)[0]
            return res.detach()
        def grad_goal(grad, vec):
            return hvp(grad, vec).unsqueeze(-1) - unlearn_grad.detach()
        
        self.p = Variable(torch.empty([unlearn_grad.shape[0],1])).cuda()
        nn.init.uniform_(self.p, -self.range, self.range)
        opt = torch.optim.Adam([self.p], lr=self.if_lr)

        k_list = [20]
        best_metric = -1
        not_change = 0
        t0 = time.time()

        for if_ep in range(self.if_epoch):
            s_time = time.time()
            opt.zero_grad()
            self.p.grad = grad_goal(total_grad, self.p)
            opt.step()
            with torch.no_grad():   
                un_ui_para_temp = un_ui_para + 1.0/data_generator.n_train * self.p.squeeze()
                e_time = time.time()
                model.embedding_user.weight.data[nei_users] = un_ui_para_temp[:u_para_num].reshape(-1, u_para.shape[-1]).data + 0
                model.embedding_item.weight.data[nei_items] = un_ui_para_temp[u_para_num:].reshape(-1, i_para.shape[-1]).data + 0

                #    
                comprehensive_ranking_metrics = compute_comprehensive_ranking_metrics(original_model, model, data_generator, top_k=10)
                current_drop_ratio = comprehensive_ranking_metrics['unlearn_positive_rank_drop_ratio']
                print("  positive interaction   :", current_drop_ratio)

                # Ranking metrics
                valid_metrics = compute_ranking_metrics(model, data_generator.valid, data_generator, k_list)
                test_metrics = compute_ranking_metrics(model, data_generator.test, data_generator, k_list)

                print(f"epoch: {if_ep}, time: {e_time-s_time:.6f}")
                for k in k_list:
                    print(f"Valid Recall@{k}: {valid_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics[f'ndcg_at_{k}']:.4f}")
                    print(f"Test  Recall@{k}: {test_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics[f'ndcg_at_{k}']:.4f}")

                #   90%   
                if baseline_valid_recall_20 is not None and valid_metrics['recall_at_20'] < 0.9 * baseline_valid_recall_20:
                    print("Valid Recall@20   90%    .")
                    break

                #     
                if current_drop_ratio > best_metric:
                    best_metric = current_drop_ratio
                    print("save best model (by rank drop ratio)")
                    torch.save(model.state_dict(), self.save_name)
                    not_change = 0
                else:
                    not_change += 1

            if not_change > 10:
                break
            print('time_cost:',time.time()-t0)
    
    def compute_neighbor_influence_clip(self, data_generator, k_hop=0):
        # full train  
        train_data = data_generator.train_original.values.copy()
        matrix_size = data_generator.n_users + data_generator.n_items
        train_data[:,1] += data_generator.n_users
        train_data[:,-1] = np.ones_like(train_data[:,-1])

        train_data2 = np.ones_like(train_data)
        train_data2[:,0] = train_data[:,1]
        train_data2[:,1] = train_data[:,0]

        paddding = np.concatenate([np.arange(matrix_size).reshape(-1,1), np.arange(matrix_size).reshape(-1,1), np.ones(matrix_size).reshape(-1,1)],axis=-1)
        data = np.concatenate([train_data, train_data2, paddding],axis=0).astype(int)
        train_matrix = sp.csc_matrix((data[:,-1],(data[:,0],data[:,1])),shape=(matrix_size,matrix_size))

        degree = np.array(train_matrix.sum(axis=-1)).squeeze()
        
        #     
        unlearn_user = data_generator.train_unlearn_original['user'].values.reshape(-1)
        unlearn_user, cunt_u = np.unique(unlearn_user, return_counts=True)
        unlearn_item = data_generator.train_unlearn_original['item'].values.reshape(-1) + data_generator.n_users
        unlearn_item, cunt_i = np.unique(unlearn_item,return_counts=True)

        unlearn_ui = np.concatenate([unlearn_user, unlearn_item], axis=-1)
        unlearn_ui_cunt = np.concatenate([cunt_u, cunt_i], axis=-1)
        degree_k = degree[unlearn_ui]
        neighbor_set = dict(zip(unlearn_ui, unlearn_ui_cunt*1.0/degree_k))
        neighbor_set_list = [neighbor_set]
        pre_neighbor_set = neighbor_set
        print("neighbor_set size:", len(neighbor_set))
        
        # k-hop 
        for i in range(k_hop):
            next_neighbor_set = dict()
            existing_nodes = list(pre_neighbor_set.keys())
            nonzero_raw, nonzero_col = train_matrix[existing_nodes].nonzero()
            for kk in range(nonzero_raw.shape[0]):
                out_node = existing_nodes[nonzero_raw[kk]]
                in_node = nonzero_col[kk]
                try:
                    next_neighbor_set[in_node] += pre_neighbor_set[out_node] * 1.0 / degree[in_node]
                except:
                    next_neighbor_set[in_node] = pre_neighbor_set[out_node] * 1.0 / degree[in_node]
            pre_neighbor_set = next_neighbor_set
            neighbor_set_list.append(next_neighbor_set)
        
        #  hop   (Book : 1-hop    )
        hop_index = 1 if k_hop >= 1 and len(neighbor_set_list) > 1 else 0
        nei_dict = neighbor_set_list[hop_index].copy()

        nei_weights = np.array(list(nei_dict.values()))
        nei_nodes = np.array(list(nei_dict.keys()))
        quantile_info = [np.quantile(nei_weights, m*0.1) for m in range(1, 11)]
        print("quantile information (median 0.1-0.2--->1): ", quantile_info)
        
        # :  40%  
        select_index = np.where(nei_weights > quantile_info[3])
        neighbor_set = nei_nodes[select_index]
        print("neighbors before filtering:", nei_nodes.shape, "after filtering:", neighbor_set.shape)

        # Book :       neighbor_set 
        all_nei_ui = neighbor_set.squeeze()
        all_nei_ui = np.unique(all_nei_ui)
        print("total influenced users+items:", all_nei_ui.shape)
        return all_nei_ui[np.where(all_nei_ui < data_generator.n_users)], all_nei_ui[np.where(all_nei_ui >= data_generator.n_users)] - data_generator.n_users


def main(config_args=None):
    args = model_hyparameters()
    assert config_args is not None
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    
    data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
    data_generator.set_train_mode(args.data_type)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    # Original LightGCN   ( )
    original_save_name = 'LightGCN_'
    original_config = {
        'lr': args.lr,
        'embed_size': args.embed_size,
        'batch_size': args.batch_size,
        'data_type': 'original',
        'dataset': args.dataset,
        'attack': args.attack,
        'seed': args.seed,
        'init_std': args.init_std,
        'gcn_layers': args.gcn_layers,
        'n_neg': args.n_neg,
    }
    for name_str, name_val in original_config.items():
        original_save_name += name_str + '-' + str(name_val) + '-'
    original_save_name = './Weights/LightGCN/' + original_save_name + 'm.pth'

    model = LightGCN(args, dataset=data_generator).cuda()
    model.load_state_dict(torch.load(original_save_name))

    # LightGCN batch_rating 
    def _batch_rating(self, users, candidated_items):
        if not isinstance(users, (list, np.ndarray)):
            users = np.array(users)
        else:
            users = np.array(users)
        if not isinstance(candidated_items, (list, np.ndarray)):
            candidated_items = np.array(candidated_items)
        else:
            candidated_items = np.array(candidated_items)
        users_t = torch.from_numpy(users).cuda().long()
        items_t = torch.from_numpy(candidated_items).cuda().long()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.computer()
            u = all_user_emb[users_t]
            i = all_item_emb[items_t]
            scores = torch.matmul(u, i.T)
        return scores.detach().cpu().numpy()
    
    model.batch_rating = types.MethodType(_batch_rating, model)

    #     batch_rating 
    original_model = LightGCN(args, dataset=data_generator).cuda()
    original_model.load_state_dict(torch.load(original_save_name))
    original_model.batch_rating = types.MethodType(_batch_rating, original_model)

    #    (Ranking)
    k_list = [20, 50, 100]
    valid_metrics = compute_ranking_metrics(model, data_generator.valid, data_generator, k_list)
    test_metrics = compute_ranking_metrics(model, data_generator.test, data_generator, k_list)

    print("***************before unlearning*************")
    for k in k_list:
        print(f"Valid Recall@{k}: {valid_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics[f'ndcg_at_{k}']:.4f}")
        print(f"Test  Recall@{k}: {test_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics[f'ndcg_at_{k}']:.4f}")
    print(config_args)

    save_name = "Weights/LightGCN_IFRU/lightgcn_dataset_{}_attack_{}_lr_{}_khop_{}_emb_{}_gcnlayers_{}.pth".format(
        args.dataset, args.attack, args.if_lr, args.k_hop, args.embed_size, args.gcn_layers)
    unlearn = influence_unlearn(save_name=save_name,if_epoch=args.if_epoch, if_lr=args.if_lr, k_hop=args.k_hop, init_range=args.if_init_std)
    unlearn.compute_hessian_with_test(model=model,data_generator=data_generator, original_model=original_model, baseline_valid_recall_20=valid_metrics['recall_at_20'])
    model.load_state_dict(torch.load(save_name))
    model.batch_rating = types.MethodType(_batch_rating, model)

    #    (Ranking)
    valid_metrics_after = compute_ranking_metrics(model, data_generator.valid, data_generator, k_list)
    test_metrics_after = compute_ranking_metrics(model, data_generator.test, data_generator, k_list)

    print("***************after unlearning***************")
    for k in k_list:
        print(f"Valid Recall@{k}: {valid_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics_after[f'ndcg_at_{k}']:.4f}")
        print(f"Test  Recall@{k}: {test_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics_after[f'ndcg_at_{k}']:.4f}")

    #   
    print("***************comprehensive ranking metrics***************")
    comprehensive_ranking_metrics = compute_comprehensive_ranking_metrics(original_model, model, data_generator, top_k=10)
    print("===   Positive Interaction Metrics ===")
    print("  positive interaction   :", comprehensive_ranking_metrics['unlearn_positive_rank_drop_ratio'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_original'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_unlearned'])
    print("  positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_rank_change'])

    # Retraining   ( )
    retraining_save_name = 'LightGCN_'
    retraining_config = {
        'lr': args.lr,
        'embed_size': args.embed_size,
        'batch_size': args.batch_size,
        'data_type': 'retraining',
        'dataset': args.dataset,
        'attack': args.attack,
        'seed': args.seed,
        'init_std': args.init_std,
        'gcn_layers': args.gcn_layers,
        'n_neg': args.n_neg,
    }
    for name_str, name_val in retraining_config.items():
        retraining_save_name += name_str + '-' + str(name_val) + '-'
    retraining_save_name = './Weights/LightGCN/' + retraining_save_name + 'm.pth'

    retraining_model = LightGCN(args, dataset=data_generator).cuda()
    try:
        retraining_model.load_state_dict(torch.load(retraining_save_name))
        retraining_model.batch_rating = types.MethodType(_batch_rating, retraining_model)
        print("Retraining   .")
    except Exception as e:
        print("Retraining    .   .")
        retraining_model = None

    if retraining_model is not None:
        print("***************retraining-based metrics***************")
        retraining_metrics = compute_retraining_based_metrics(retraining_model, model, data_generator, k_list=k_list)
        for k in k_list:
            print(f"Recall@{k}: {retraining_metrics[f'recall_at_{k}']:.4f}")
            print(f"NDCG@{k}: {retraining_metrics[f'ndcg_at_{k}']:.4f}")


if __name__=='__main__':
    config_args = {}
    config_args['embed_size'] = 48
    config_args['batch_size'] = 2048
    config_args['epoch'] = 5000
    config_args['data_path'] = 'Data/Process/'
    config_args['dataset'] = 'Amazon_Book'
    config_args['attack'] = '0.01'
    config_args['k_hop'] = 0
    config_args['data_type'] = 'full'
    config_args['if_epoch'] = 5000
    config_args['if_lr'] = 1e4
    config_args['if_init_std'] = 0
    config_args['seed'] = 1024
    config_args['lr'] = 1e-3
    config_args['regs'] = 0
    config_args['init_std'] = 1e-3
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str)
    parser.add_argument('--gcn_layers', type=int)
    _args = parser.parse_args()
    if _args.attack is not None:
        config_args['attack'] = _args.attack
    if getattr(_args, 'gcn_layers', None) is not None:
        config_args['gcn_layers'] = _args.gcn_layers
    main(config_args) 