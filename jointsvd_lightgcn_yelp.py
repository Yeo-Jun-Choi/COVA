import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
from time import time

from utility.load_data import *
from Model.Lightgcn import LightGCN


def compute_ranking_metrics_lightgcn(model, dataset, data_generator, k_list=[20, 50, 100]):
    """
    LightGCN   NDCG@K, Recall@K  

    Args:
        model:  LightGCN 
        dataset:   (valid  test)
        data_generator:  
        k_list:  K  

    Returns:
        dict: Recall@K, NDCG@K 
    """
    metrics = {}
    max_k = max(k_list)

    for k in k_list:
        metrics[f'recall_at_{k}'] = []
        metrics[f'ndcg_at_{k}'] = []

    positive_interactions = {}
    for user_id, item_id, label in dataset[['user', 'item', 'label']].values:
        if label == 1:
            if user_id not in positive_interactions:
                positive_interactions[user_id] = set()
            positive_interactions[user_id].add(item_id)

    model.eval()
    with torch.no_grad():
        all_user_emb, all_item_emb = model.computer()

    for user_id in range(data_generator.n_users):
        if user_id not in positive_interactions or len(positive_interactions[user_id]) == 0:
            continue

        #      (inner product, sigmoid :  )
        user_vec = all_user_emb[user_id]
        user_scores = torch.matmul(user_vec, all_item_emb.t()).detach().cpu().numpy()

        top_max_k_indices = np.argsort(-user_scores)[:max_k]
        top_max_k_items = [i for i in top_max_k_indices]

        user_positive_items = positive_interactions[user_id]

        for k in k_list:
            top_k_items = top_max_k_items[:k]

            relevant_items = user_positive_items
            recommended_items = set(top_k_items)
            recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items) if len(relevant_items) > 0 else 0
            metrics[f'recall_at_{k}'].append(recall)

            dcg = 0
            idcg = 0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1 / np.log2(i + 2)
            for i in range(min(k, len(relevant_items))):
                idcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg_at_{k}'].append(ndcg)

    for k in k_list:
        if len(metrics[f'recall_at_{k}']) > 0:
            metrics[f'recall_at_{k}'] = np.mean(metrics[f'recall_at_{k}'])
            metrics[f'ndcg_at_{k}'] = np.mean(metrics[f'ndcg_at_{k}'])
        else:
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0

    return metrics


def compute_comprehensive_ranking_metrics_lightgcn(original_model, unlearned_model, data_generator):
    """
    LightGCN    (  positive interactions )
    """
    unlearn_interactions = data_generator.train_random[['user', 'item', 'label']].values

    ranking_metrics = {
        'unlearn_positive_rank_drop_ratio': 0.0,
        'unlearn_positive_avg_rank_original': 0.0,
        'unlearn_positive_avg_rank_unlearned': 0.0,
        'unlearn_positive_rank_change': 0.0,
    }

    if len(unlearn_interactions) == 0:
        return ranking_metrics

    user_interactions = {}
    for user_id, item_id in data_generator.train[['user', 'item']].values:
        if user_id not in user_interactions:
            user_interactions[user_id] = set()
        user_interactions[user_id].add(item_id)

    user_unlearn_items = {}
    for user_id, item_id, label in unlearn_interactions:
        if user_id not in user_interactions:
            continue
        if user_id not in user_unlearn_items:
            user_unlearn_items[user_id] = []
        user_unlearn_items[user_id].append(item_id)

    with torch.no_grad():
        orig_user_emb, orig_item_emb = original_model.computer()
        un_user_emb, un_item_emb = unlearned_model.computer()

    un_pos_orig_ranks = []
    un_pos_unl_ranks = []
    rank_drop_count = 0

    for user_id, un_items in user_unlearn_items.items():
        if not un_items:
            continue
        user_items = list(user_interactions[user_id])
        item_to_idx = {it: idx for idx, it in enumerate(user_items)}

        uvec_o = orig_user_emb[user_id]
        uvec_u = un_user_emb[user_id]
        item_mat_o = orig_item_emb[user_items]
        item_mat_u = un_item_emb[user_items]

        scores_o = torch.matmul(uvec_o, item_mat_o.t()).detach().cpu().numpy()
        scores_u = torch.matmul(uvec_u, item_mat_u.t()).detach().cpu().numpy()

        ranks_o = np.argsort(np.argsort(-scores_o)) + 1
        ranks_u = np.argsort(np.argsort(-scores_u)) + 1

        for item in un_items:
            if item not in item_to_idx:
                continue
            idx = item_to_idx[item]
            r_o = ranks_o[idx]
            r_u = ranks_u[idx]
            un_pos_orig_ranks.append(r_o)
            un_pos_unl_ranks.append(r_u)
            if r_o - r_u < 0:
                rank_drop_count += 1

    if un_pos_orig_ranks:
        ranking_metrics['unlearn_positive_rank_drop_ratio'] = rank_drop_count / len(un_pos_orig_ranks)
        ranking_metrics['unlearn_positive_avg_rank_original'] = float(np.mean(un_pos_orig_ranks))
        ranking_metrics['unlearn_positive_avg_rank_unlearned'] = float(np.mean(un_pos_unl_ranks))
        ranking_metrics['unlearn_positive_rank_change'] = float(np.mean(un_pos_orig_ranks) - np.mean(un_pos_unl_ranks))

    ranking_metrics['overall_effectiveness'] = ranking_metrics['unlearn_positive_rank_drop_ratio']
    return ranking_metrics


def compute_retraining_based_metrics_lightgcn(retraining_model, unlearning_model, data_generator, k_list=[20, 50, 100]):
    """
    Retraining LightGCN  Unlearning LightGCN Recall@K, NDCG@K 
    (   )
    """
    metrics = {}
    max_k = max(k_list)
    for k in k_list:
        metrics[f'recall_at_{k}'] = []
        metrics[f'ndcg_at_{k}'] = []

    user_train_items = {}
    for user_id, item_id in data_generator.train_normal[['user', 'item']].values:
        if user_id not in user_train_items:
            user_train_items[user_id] = set()
        user_train_items[user_id].add(item_id)

    retraining_model.eval()
    unlearning_model.eval()
    with torch.no_grad():
        rt_u, rt_i = retraining_model.computer()
        ul_u, ul_i = unlearning_model.computer()

    for user_id in range(data_generator.n_users):
        all_items = np.arange(data_generator.n_items)

        r_user_vec = rt_u[user_id]
        u_user_vec = ul_u[user_id]
        r_scores = torch.matmul(r_user_vec, rt_i.t()).detach().cpu().numpy()
        u_scores = torch.matmul(u_user_vec, ul_i.t()).detach().cpu().numpy()

        train_items = user_train_items.get(user_id, set())
        if len(train_items) > 0:
            r_scores[list(train_items)] = -np.inf
            u_scores[list(train_items)] = -np.inf

        r_top_idx = np.argsort(-r_scores)[:max_k]
        u_top_idx = np.argsort(-u_scores)[:max_k]

        r_top_items = [i for i in r_top_idx]
        u_top_items = [i for i in u_top_idx]

        for k in k_list:
            r_k = set(r_top_items[:k])
            u_k = set(u_top_items[:k])
            recall = len(r_k.intersection(u_k)) / len(r_k) if len(r_k) > 0 else 0
            metrics[f'recall_at_{k}'].append(recall)

            dcg = 0
            idcg = 0
            for i, item in enumerate(u_top_items[:k]):
                if item in r_k:
                    dcg += 1 / np.log2(i + 2)
            for i in range(min(k, len(r_k))):
                idcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg_at_{k}'].append(ndcg)

    for k in k_list:
        metrics[f'recall_at_{k}'] = float(np.mean(metrics[f'recall_at_{k}'])) if len(metrics[f'recall_at_{k}']) > 0 else 0.0
        metrics[f'ndcg_at_{k}'] = float(np.mean(metrics[f'ndcg_at_{k}'])) if len(metrics[f'ndcg_at_{k}']) > 0 else 0.0

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
        self.data_type = 'full'
        self.init_std = 1e-4
        self.seed = 1024
        self.n_neg = 1

        # LightGCN hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0

        # Joint SVD params
        self.alpha = 0.5
        self.beta = 0.5
        self.w1 = 1.0
        self.w2 = 1.0
        self.w3 = 1.0
        self.num_iter = 10

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class joint_svd_unlearn(nn.Module):
    def __init__(self, save_name, alpha=0.5, beta=0.5, w1=1.0, w2=1.0, w3=1.0, num_iter=10, num_dim=48) -> None:
        super(joint_svd_unlearn, self).__init__()
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
        Joint SVD    (Yelp + LightGCN)
        """
        original_interaction_matrix = torch.zeros(data_generator.n_users, data_generator.n_items)

        train_normal_data = data_generator.train_normal[['user', 'item', 'label']].values
        for user_id, item_id, label in train_normal_data:
            if label == 1:
                original_interaction_matrix[user_id, item_id] = 1

        train_random_data = data_generator.train_random[['user', 'item', 'label']].values
        for user_id, item_id, label in train_random_data:
            if label == 0:
                original_interaction_matrix[user_id, item_id] = 1

        ideal_interaction_matrix = original_interaction_matrix.clone()
        unlearn_data = data_generator.train_random[['user', 'item']].values
        for user_id, item_id in unlearn_data:
            ideal_interaction_matrix[user_id, item_id] = 0

        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.computer()
            predict_prob = torch.matmul(user_emb, item_emb.t()).detach().cpu()

        user_emb_u, item_emb_u = joint_svd_unlearning(
            original_interaction_matrix,
            ideal_interaction_matrix,
            predict_prob,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            beta=self.beta,
            num_iter=self.num_iter,
            num_dim=self.num_dim,
            alpha=self.alpha
        )

        #   (LightGCN naming)
        model.embedding_user.weight.data = user_emb_u.cuda() if user_emb_u.device.type == 'cpu' else user_emb_u
        model.embedding_item.weight.data = item_emb_u.cuda() if item_emb_u.device.type == 'cpu' else item_emb_u
        model.n_layers=0
        torch.save(model.state_dict(), self.save_name)
        return model


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
    data_generator.set_train_mode(mode=args.data_type)

    # LightGCN      (original)
    model = LightGCN(args, dataset=data_generator).cuda()

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

    model.load_state_dict(torch.load(original_save_name))

    #   
    original_model = LightGCN(args, dataset=data_generator).cuda()
    original_model.load_state_dict(torch.load(original_save_name))

    # retraining   (optional)
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
        print('Retraining   ')
    except Exception as e:
        print(f'Retraining   : {e}')
        retraining_model = None

    #   
    k_list = [20, 50, 100]
    valid_metrics = compute_ranking_metrics_lightgcn(model, data_generator.valid, data_generator, k_list)
    test_metrics = compute_ranking_metrics_lightgcn(model, data_generator.test, data_generator, k_list)

    print('***************before unlearning*************')
    for k in k_list:
        print(f"Valid Recall@{k}: {valid_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics[f'ndcg_at_{k}']:.4f}")
        print(f"Test  Recall@{k}: {test_metrics[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics[f'ndcg_at_{k}']:.4f}")
# Joint SVD 
    save_name = "Weights/LightGCN_JointSVD/lightgcn_dataset_{}_attack_{}_alpha_{}_beta_{}_w1_{}_w2_{}_w3_{}_gcnlayers_{}.pth".format(
        args.dataset, args.attack, args.alpha, args.beta, args.w1, args.w2, args.w3, getattr(args, 'gcn_layers', 1)
    )
    unlearner = joint_svd_unlearn(
        save_name=save_name,
        alpha=args.alpha,
        beta=args.beta,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        num_iter=args.num_iter,
        num_dim=args.embed_size,
    )
    model = unlearner.compute_joint_svd_unlearning(model=model, data_generator=data_generator, args=args)

    #   
    valid_metrics_after = compute_ranking_metrics_lightgcn(model, data_generator.valid, data_generator, k_list)
    test_metrics_after = compute_ranking_metrics_lightgcn(model, data_generator.test, data_generator, k_list)

    print('***************after unlearning***************')
    for k in k_list:
        print(f"Valid Recall@{k}: {valid_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics_after[f'ndcg_at_{k}']:.4f}")
        print(f"Test  Recall@{k}: {test_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics_after[f'ndcg_at_{k}']:.4f}")
######################## ########################
    print("***************Request from Professor***************")
    comprehensive_ranking_metrics = compute_comprehensive_ranking_metrics_lightgcn(original_model, retraining_model, data_generator)
    
    #   positive interaction  
    print("===   Positive Interaction Metrics ===")
    print("  positive interaction   :", comprehensive_ranking_metrics['unlearn_positive_rank_drop_ratio'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_original'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_unlearned'])
    print("  positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_rank_change'])


    #    (  positive )
    print('***************comprehensive ranking metrics***************')
    comp_metrics = compute_comprehensive_ranking_metrics_lightgcn(original_model, model, data_generator)
    print("===   Positive Interaction Metrics ===")
    print("  positive interaction   :", comp_metrics['unlearn_positive_rank_drop_ratio'])
    print("   :", comp_metrics['unlearn_positive_avg_rank_original'])
    print("   :", comp_metrics['unlearn_positive_avg_rank_unlearned'])
    print(" :", comp_metrics['unlearn_positive_rank_change'])
# Retraining   (optional)
    if retraining_model is not None:
        print('***************retraining-based metrics***************')
        rt_metrics = compute_retraining_based_metrics_lightgcn(retraining_model, model, data_generator, k_list=k_list)
        for k in k_list:
            print(f"Recall@{k}: {rt_metrics[f'recall_at_{k}']:.4f}")
            print(f"NDCG@{k}: {rt_metrics[f'ndcg_at_{k}']:.4f}")
print('***************retraining-based metrics to original model***************')
        rt2 = compute_retraining_based_metrics_lightgcn(retraining_model, original_model, data_generator, k_list=k_list)
        for k in k_list:
            print(f"Recall@{k}: {rt2[f'recall_at_{k}']:.4f}")
            print(f"NDCG@{k}: {rt2[f'ndcg_at_{k}']:.4f}")
else:
        print('Retraining   ground truth   .')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint SVD Unlearning Experiment for Yelp with LightGCN')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--w1', type=float, default=1.0)
    parser.add_argument('--w2', type=float, default=1.0)
    parser.add_argument('--w3', type=float, default=1.0)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--embed_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--dataset', type=str, default='Yelp')
    parser.add_argument('--attack', type=str, default='0.01')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_std', type=float, default=1e-3)
    parser.add_argument('--version', type=str, default='mean')
    parser.add_argument('--n_neg', type=int, default=1)
    parser.add_argument('--gcn_layers', type=int, default=1)

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
    if getattr(args_cmd, 'gcn_layers', None) is not None:
        config_args['gcn_layers'] = args_cmd.gcn_layers
    config_args['init_std'] = args_cmd.init_std
    config_args['alpha'] = args_cmd.alpha
    config_args['beta'] = args_cmd.beta
    config_args['w1'] = args_cmd.w1
    config_args['w2'] = args_cmd.w2
    config_args['w3'] = args_cmd.w3
    config_args['num_iter'] = args_cmd.num_iter
    config_args['script_name'] = 'jointsvd_lightgcn_yelp.py'
    config_args['version'] = args_cmd.version
    config_args['n_neg'] = args_cmd.n_neg

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