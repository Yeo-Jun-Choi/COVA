import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from utility.load_data import * 
import scipy.sparse as sp
import torch.nn.functional as F
from Model.MF import MF
import time
from sklearn.metrics import roc_auc_score
import time
from torch.autograd import Variable
from utility.compute import *
import random
import argparse


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'Yelp'
        self.attack = '0.01'
        self.k_hop = 0
        self.data_type = 'full'
        self.if_epoch = 5000
        self.if_lr = 1e4
        self.if_init_std = 0
        self.seed = 1024
        self.lr = 1e-3
        self.regs = 0
        self.init_std = 0
        self.alpha = 0.5  # Joint SVD 
        self.beta = 0.5   # Joint SVD 
        self.stacking_method = 'vertical'  # 'vertical', 'horizontal', 'weighted_sum'
        self.num_iter = 10
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)


class joint_svd_unlearn(nn.Module):
    def __init__(self, save_name, alpha=0.5, beta=0.5, w1=1.0, w2=2.0, w3=1.5, num_iter=5, num_dim=48) -> None:
        super(joint_svd_unlearn).__init__()
        self.alpha = alpha
        self.beta = beta
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.save_name = save_name
        self.num_iter = num_iter
        self.num_dim = num_dim

    def compute_joint_svd_unlearning(self, model=None, data_generator=None, args=None):
        """
        Joint SVD    (Yelp )
        """
        # 1.     (1: positive, 0: unobserved)
        original_interaction_matrix = torch.zeros(data_generator.n_users, data_generator.n_items)
        
        # Positive interactions (train_normal label=1 )
        train_normal_data = data_generator.train_normal[['user', 'item', 'label']].values
        for user_id, item_id, label in train_normal_data:
            if label == 1:  # positive interaction
                original_interaction_matrix[user_id, item_id] = 1
        
        #   positive interactions (train_random label=0  -  positive)
        train_random_data = data_generator.train_random[['user', 'item', 'label']].values
        for user_id, item_id, label in train_random_data:
            if label == 0:  #   positive interaction
                original_interaction_matrix[user_id, item_id] = 1

        # 2.     (  interaction )
        ideal_interaction_matrix = original_interaction_matrix.clone()
        unlearn_data = data_generator.train_random[['user', 'item']].values
        for user_id, item_id in unlearn_data:
            #   interaction unobserved(0) 
            ideal_interaction_matrix[user_id, item_id] = 0

        # 3.     
        model.eval()
        with torch.no_grad():
            all_users = np.arange(data_generator.n_users)
            all_items = np.arange(data_generator.n_items)
            predict_prob = torch.from_numpy(model.batch_rating(all_users, all_items)).cpu()

        # 4. Joint SVD  
        user_emb, item_emb = joint_svd_unlearning(
            original_interaction_matrix,
            ideal_interaction_matrix,
            predict_prob,
            w1 = self.w1,
            w2 = self.w2,
            w3 = self.w3,
            beta = self.beta,
            num_iter = self.num_iter,
            num_dim = self.num_dim,
            alpha = self.alpha
        )

        # 5.    
        self.integrate_unlearned_matrix_to_embeddings(model, user_emb, item_emb, data_generator)

        #  
        torch.save(model.state_dict(), self.save_name)
        return model

    def integrate_unlearned_matrix_to_embeddings(self, model, user_emb, item_emb, data_generator):
        """
            
        """
        model.user_embeddings.weight.data = user_emb
        model.item_embeddings.weight.data = item_emb


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
    
    data_generator = Data_for_MF(data_path=args.data_path + args.dataset + '/' + args.attack, batch_size=args.batch_size)
    data_generator.set_train_mode(args.data_type)
    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    # Original model  
    original_save_name = 'MF_'
    original_config = {
        'lr': args.lr, 'embed_size': args.embed_size, 'batch_size': args.batch_size,
        'data_type': 'original', 'dataset': args.dataset, 'attack': args.attack,
        'seed': args.seed, 'init_std': args.init_std, 'n_neg': 1
    }
    for name_str, name_val in original_config.items():
        original_save_name += name_str + '-' + str(name_val) + '-'
    original_save_name = './Weights/MF/' + original_save_name + "m.pth"
    
    model = MF(data_config=config,args=args).cuda()
    model.load_state_dict(torch.load(original_save_name))

    #   
    original_model = MF(data_config=config,args=args).cuda()
    original_model.load_state_dict(torch.load(original_save_name))

    # Retraining   (ground truth)
    retraining_save_name = 'MF_'
    retraining_config = {
        'lr': args.lr, 'embed_size': args.embed_size, 'batch_size': args.batch_size,
        'data_type': 'retraining', 'dataset': args.dataset, 'attack': args.attack,
        'seed': args.seed, 'init_std': args.init_std, 'n_neg': 1
    }
    for name_str, name_val in retraining_config.items():
        retraining_save_name += name_str + '-' + str(name_val) + '-'
    retraining_save_name = './Weights/MF/' + retraining_save_name + "m.pth"
    
    retraining_model = MF(data_config=config,args=args).cuda()
    try:
        retraining_model.load_state_dict(torch.load(retraining_save_name))
        print("Retraining   .")
    except:
        print("Retraining    .   .")
        retraining_model = None

    # Ranking metrics 
    k_list = [20, 50, 100]
    test_metrics = compute_ranking_metrics(model, data_generator.test, data_generator, k_list)
    valid_metrics = compute_ranking_metrics(model, data_generator.valid, data_generator, k_list)

    print("***************before unlearning*************")
    for k in k_list:
        print(f"Valid Recall@{k}: {valid_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics[f'ndcg_at_{k}']:.4f}")
        print(f"Test Recall@{k}: {test_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics[f'ndcg_at_{k}']:.4f}")
save_name = "Weights/MF_JointSVD/mf_dataset_{}_attack_{}_alpha_{}_beta_{}_w1_{}_w2_{}_w3_{}.pth".format(
        args.dataset, args.attack, args.alpha, args.beta, args.w1, args.w2, args.w3
    )
    unlearn = joint_svd_unlearn(
        save_name=save_name,
        alpha=args.alpha,
        beta=args.beta,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        num_iter=args.num_iter,
        num_dim=args.embed_size
    )
    model = unlearn.compute_joint_svd_unlearning(model=model, data_generator=data_generator, args=args)

    # Ranking metrics  (after unlearning)
    test_metrics_after = compute_ranking_metrics(model, data_generator.test, data_generator, k_list)
    valid_metrics_after = compute_ranking_metrics(model, data_generator.valid, data_generator, k_list)
    
    print("***************after unlearning***************")
    for k in k_list:
        print(f"Valid Recall@{k}: {valid_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics_after[f'ndcg_at_{k}']:.4f}")
        print(f"Test Recall@{k}: {test_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics_after[f'ndcg_at_{k}']:.4f}")
######################## ########################
    print("***************Request from Professor***************")
    comprehensive_ranking_metrics = compute_comprehensive_ranking_metrics(original_model, retraining_model, data_generator, top_k=10)
    
    #   positive interaction  
    print("===   Positive Interaction Metrics ===")
    print("  positive interaction   :", comprehensive_ranking_metrics['unlearn_positive_rank_drop_ratio'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_original'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_unlearned'])
    print("  positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_rank_change'])
    
##################################################################

    #    
    print("***************comprehensive ranking metrics***************")
    comprehensive_ranking_metrics = compute_comprehensive_ranking_metrics(original_model, model, data_generator, top_k=10)
    
    #   positive interaction  
    print("===   Positive Interaction Metrics ===")
    print("  positive interaction   :", comprehensive_ranking_metrics['unlearn_positive_rank_drop_ratio'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_original'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_unlearned'])
    print("  positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_rank_change'])
    
    # Wandb 
k_list = [20, 50, 100]
    # Retraining  
    if retraining_model is not None:
        print("***************retraining-based metrics***************")
        retraining_metrics = compute_retraining_based_metrics(retraining_model, model, data_generator, k_list=k_list)
        for k in k_list:
            print(f"Recall@{k}: {retraining_metrics[f'recall_at_{k}']:.4f}")
            print(f"NDCG@{k}: {retraining_metrics[f'ndcg_at_{k}']:.4f}")
print("***************retraining-based metrics to original model***************")
        retraining_metrics = compute_retraining_based_metrics(retraining_model, original_model, data_generator, k_list=k_list)
        for k in k_list:
            print(f"Recall@{k}: {retraining_metrics[f'recall_at_{k}']:.4f}")
            print(f"NDCG@{k}: {retraining_metrics[f'ndcg_at_{k}']:.4f}")
else:
        print("Retraining   ground truth   .")


if __name__=='__main__':
    #   
    parser = argparse.ArgumentParser(description='Joint SVD Unlearning Experiment for Yelp')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for Joint SVD')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for Joint SVD')
    parser.add_argument('--w1', type=float, default=1.0, help='w1 parameter for Joint SVD')
    parser.add_argument('--w2', type=float, default=1.0, help='w2 parameter for Joint SVD')
    parser.add_argument('--w3', type=float, default=1.0, help='w3 parameter for Joint SVD')
    parser.add_argument('--num_iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('--embed_size', type=int, default=48, help='Embedding size')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--dataset', type=str, default='Yelp', help='Dataset name')
    parser.add_argument('--attack', type=str, default='0.01', help='Attack ratio')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--init_std', type=float, default=1e-3, help='Initialization standard deviation')
    parser.add_argument('--version', type=str, default='mean', help='Version of Joint SVD')
    
    args_cmd = parser.parse_args()
    
    config_args = {}
    config_args['embed_size'] = args_cmd.embed_size
    config_args['batch_size'] = args_cmd.batch_size
    config_args['epoch'] = 5000
    config_args['data_path'] = 'Data/Process/'
    config_args['dataset'] = args_cmd.dataset
    config_args['attack'] = args_cmd.attack
    config_args['k_hop'] = 0
    config_args['data_type'] = 'original'
    config_args['if_epoch'] = 5000
    config_args['if_lr'] = 1e4
    config_args['if_init_std'] = 0
    config_args['seed'] = args_cmd.seed
    config_args['lr'] = args_cmd.lr
    config_args['regs'] = 0
    config_args['init_std'] = args_cmd.init_std
    config_args['alpha'] = args_cmd.alpha
    config_args['beta'] = args_cmd.beta
    config_args['w1'] = args_cmd.w1
    config_args['w2'] = args_cmd.w2
    config_args['w3'] = args_cmd.w3
    config_args['num_iter'] = args_cmd.num_iter
    
    config_args['script_name'] = "jointsvd_mf_yelp.py"
    config_args['version'] = args_cmd.version
    
    if config_args['version'] == 'mean':
        from joint_svd import joint_svd_unlearning_v3_mean as joint_svd_unlearning
    elif config_args['version'] == 'minmax':
        from joint_svd import joint_svd_unlearning_v3_minmax as joint_svd_unlearning
    elif config_args['version'] == 'gaussian':
        from joint_svd import joint_svd_unlearning_v3_gaussian as joint_svd_unlearning
    elif config_args['version'] == 'mean_user_weight':
        from joint_svd import joint_svd_unlearning_v3_mean_user_weight as joint_svd_unlearning
    elif config_args['version'] == 'minmax_user_weight':
        from joint_svd import joint_svd_unlearning_v3_minmax_user_weight as joint_svd_unlearning
    elif config_args['version'] == 'gaussian_user_weight':
        from joint_svd import joint_svd_unlearning_v3_gaussian_user_weight as joint_svd_unlearning
    elif config_args['version'] == 'mean2':
        from joint_svd import joint_svd_unlearning_v3_mean2 as joint_svd_unlearning
    elif config_args['version'] == 'mean3':
        from joint_svd import joint_svd_unlearning_v3_mean3 as joint_svd_unlearning
    elif config_args['version'] == 'mean4':
        from joint_svd import joint_svd_unlearning_v3_mean4 as joint_svd_unlearning
    elif config_args['version'] == 'v4':
        from joint_svd import joint_svd_unlearning_v4 as joint_svd_unlearning
    elif config_args['version'] == 'v4_wonorm':
        from joint_svd import joint_svd_unlearning_v4_wonorm as joint_svd_unlearning
    elif config_args['version'] == 'v4_min':
        from joint_svd import joint_svd_unlearning_v4_min as joint_svd_unlearning
    elif config_args['version'] == 'v4_min_wonorm':
        from joint_svd import joint_svd_unlearning_v4_min_wonorm as joint_svd_unlearning
    elif config_args['version'] == 'v5':
        from joint_svd import joint_svd_unlearning_v5 as joint_svd_unlearning
    elif config_args['version'] == 'v6':
        from joint_svd import joint_svd_unlearning_v6 as joint_svd_unlearning
    elif config_args['version'] == 'v7':
        from joint_svd import joint_svd_unlearning_v7 as joint_svd_unlearning
    elif config_args['version'] == 'v8':
        from joint_svd import joint_svd_unlearning_v8 as joint_svd_unlearning
    elif config_args['version'] == 'v9':
        from joint_svd import joint_svd_unlearning_v9 as joint_svd_unlearning
    main(config_args) 