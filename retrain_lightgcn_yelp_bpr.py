from distutils.command.config import config
import random
import torch
import numpy as np
import torch.optim
from time import time
import os
from utility.load_data import *
from sklearn.metrics import roc_auc_score
from Model.Lightgcn import LightGCN
from utility.compute import *
from utility.negative_sampling import sample_triplets_for_bpr


def compute_ranking_metrics_lightgcn(model, test_data, data_generator, k_list=[20, 50, 100]):
    """
    LightGCN   ranking metrics  
    
    Args:
        model: LightGCN 
        test_data:  
        data_generator:  
        k_list:  K  
    
    Returns:
        dict: Recall@K, NDCG@K 
    """
    metrics = {}
    max_k = max(k_list)
    
    # metrics   
    for k in k_list:
        metrics[f'recall_at_{k}'] = []
        metrics[f'ndcg_at_{k}'] = []
    
    # Positive interaction   (label 1 )
    positive_interactions = {}
    for user_id, item_id, label in test_data[['user', 'item', 'label']].values:
        if label == 1:  # positive interaction
            if user_id not in positive_interactions:
                positive_interactions[user_id] = set()
            positive_interactions[user_id].add(item_id)
    
    #    
    for user_id in range(data_generator.n_users):
        if user_id not in positive_interactions or len(positive_interactions[user_id]) == 0:
            continue
            
        # LightGCN  getUsersRating  
        user_tensor = torch.tensor([user_id]).cuda()
        user_scores = model.getUsersRating(user_tensor).detach().cpu().numpy().flatten()
        
        # Top-max_k 
        top_max_k_indices = np.argsort(-user_scores)[:max_k]
        top_max_k_items = [i for i in top_max_k_indices]
        
        #   positive 
        user_positive_items = positive_interactions[user_id]
        
        #  K  
        for k in k_list:
            # Top-k  
            top_k_items = top_max_k_items[:k]
            
            # Recall@K 
            relevant_items = user_positive_items
            recommended_items = set(top_k_items)
            recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items) if len(relevant_items) > 0 else 0
            metrics[f'recall_at_{k}'].append(recall)
            
            # NDCG@K 
            dcg = 0
            idcg = 0
            
            # DCG 
            for i, item_id in enumerate(top_k_items):
                if item_id in relevant_items:
                    dcg += 1 / np.log2(i + 2)  # log2(i+2) because i starts from 0
            
            # IDCG  (ideal ranking)
            for i in range(min(k, len(relevant_items))):
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg_at_{k}'].append(ndcg)
    
    #  
    for k in k_list:
        if len(metrics[f'recall_at_{k}']) > 0:
            metrics[f'recall_at_{k}'] = np.mean(metrics[f'recall_at_{k}'])
            metrics[f'ndcg_at_{k}'] = np.mean(metrics[f'ndcg_at_{k}'])
        else:
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
    
    return metrics


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'Yelp'
        self.attack = '0.01'
        self.layer_size = '[64,64]'
        self.verbose = 1
        self.Ks = '[10]'
        self.data_type = 'retraining'  # retraining  
        self.init_std = 1e-4
        self.seed = 1024
        self.n_neg = 1  # negative sample 

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class early_stoper(object):
    def __init__(self, refer_metric='valid_recall_20', stop_condition=10):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric

    def update_and_isbest(self, eval_metric, epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        elif eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
            self.best_eval_result = eval_metric
            self.not_change = 0
            self.best_epoch = epoch
            return True
        else:
            self.not_change += 1
            return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False


def main(config_args):
    args = model_hyparameters()
    assert config_args is not None
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_name = 'LightGCN_'
    for name_str, name_val in config_args.items():
        save_name += name_str + '-' + str(name_val) + '-'

    data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
    data_generator.set_train_mode(mode=args.data_type)  # retraining  

    model = LightGCN(args, dataset=data_generator).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = 0
    best_valid_recall = 0
    best_test_recall = 0
    e_stoper = early_stoper(refer_metric='valid_recall_20', stop_condition=10)
    print("Start training with retraining mode")
    
    for epoch in range(args.epoch):
        t1 = time()
        loss, bpr_loss, reg_loss = 0., 0., 0.
        
        # Positive   BPR  (retraining )
        pos_data = data_generator.train[data_generator.train['label'] == 1][['user', 'item']].values
        
        # Batch   ( batch negative sampling)
        n_batches = len(pos_data) // args.batch_size + (1 if len(pos_data) % args.batch_size != 0 else 0)
        
        for i in range(n_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(pos_data))
            
            batch_pos_data = pos_data[start_idx:end_idx]
            
            #  batch negative sampling
            users = batch_pos_data[:, 0].astype(np.int64)
            pos_items = batch_pos_data[:, 1].astype(np.int64)
            
            #  negative sampling ( )
            neg_items = np.random.randint(0, data_generator.n_items, size=(len(users), args.n_neg))
            
            batch_users = torch.from_numpy(users).cuda().long()
            batch_pos_items = torch.from_numpy(pos_items).cuda().long()
            batch_neg_items = torch.from_numpy(neg_items).cuda().long()
            
            # LightGCN  BPR loss 
            opt.zero_grad()
            batch_bpr_loss = model.compute_bpr_loss(batch_users, batch_pos_items, batch_neg_items)
            batch_bpr_loss.backward()
            opt.step()
            
            loss += batch_bpr_loss.item()
            bpr_loss += batch_bpr_loss.item()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        # Ranking metrics 
        k_list = [20, 50, 100]
        # LightGCN     
        valid_metrics = compute_ranking_metrics_lightgcn(model, data_generator.valid, data_generator, k_list)
        test_metrics = compute_ranking_metrics_lightgcn(model, data_generator.test, data_generator, k_list)

        t3 = time()
        perf_str = "epoch: %d, time: %.6f, bpr_loss:%.6f" %(epoch, t3-t2, bpr_loss)
        print(perf_str)
        
        # Ranking metrics 
        for k in k_list:
            print(f"Valid Recall@{k}: {valid_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics[f'ndcg_at_{k}']:.4f}")
            print(f"Test Recall@{k}: {test_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics[f'ndcg_at_{k}']:.4f}")

        one_result = {'valid_recall_20': valid_metrics['recall_at_20'], 'test_recall_20': test_metrics['recall_at_20']}
        is_best = e_stoper.update_and_isbest(one_result, epoch)
        if is_best:
            best_epoch = epoch
            best_valid_recall = valid_metrics['recall_at_20']
            best_test_recall = test_metrics['recall_at_20']
            torch.save(model.state_dict(), './Weights/LightGCN/' + save_name + "m.pth")
            print("saving the best model")

        if e_stoper.is_stop():
            print("save path for best model:", './Weights/LightGCN/' + save_name + "m.pth")
            break

    final_perf = 'best_epoch = {}, best_valid_recall@20 = {}, best_test_recall@20 = {}'.format(best_epoch, best_valid_recall, best_test_recall)
    print(final_perf)


if __name__ == '__main__':
    config = {
        'lr': 1e-3,  # [1e-2, 1e-3, 1e-4]
        'embed_size': 48,  # [32, 48, 64]
        'batch_size': 2048,
        'data_type': 'retraining',  # retraining  
        'dataset': 'Yelp',  #[Yelp, Gowalla, Amazon_Book]
        'attack':'0.01',  # [0.02, 0.01]
        'seed': 1024,
        'init_std': 1e-3,  # [1e-2, 1e-3, 1e-4]
        'gcn_layers': 1,
        'n_neg': 1  # negative sample 
    }
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str)
    parser.add_argument('--gcn_layers', type=int)
    _args = parser.parse_args()
    if _args.attack is not None:
        config['attack'] = _args.attack
    if getattr(_args, 'gcn_layers', None) is not None:
        config['gcn_layers'] = _args.gcn_layers
    main(config) 