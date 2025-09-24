import numpy as np
from utility.load_data import *
import os
import sys
import pickle
import torch
import glob
from time import time
import random
from Model.Eraser import RecEraser_LightGCN
from Model.Lightgcn import LightGCN
from utility.compute import compute_ranking_metrics_sisa, compute_ranking_metrics, compute_comprehensive_ranking_metrics, compute_retraining_based_metrics_sisa



def compute_comprehensive_ranking_metrics_lightgcn(original_model, unlearned_model, data_generator, top_k=None):
    """
    LightGCN    (  positive interactions )
    - original_model: Original LightGCN (computer() )
    - unlearned_model: RecEraser_LightGCN .        
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
    for user_id, item_id in data_generator.train_original[['user', 'item']].values:
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

        # Unlearned  : RecEraser_LightGCN    
        if getattr(unlearned_model, 'model_type', '') == 'RecEraser_LightGCN':
            users_emb = unlearned_model.user_embedding.weight  # [n_users, num_local*emb_dim]
            items_emb = unlearned_model.item_embedding.weight  # [n_items, num_local*emb_dim]
            emb_dim = unlearned_model.emb_dim
            num_local = unlearned_model.num_local

            user_shard_final = []  #    LightGCN  (n_users, emb_dim)
            item_shard_final = []  #    LightGCN  (n_items, emb_dim)
            for i in range(num_local):
                temp_u = users_emb[:, i * emb_dim:(i + 1) * emb_dim]
                temp_i = items_emb[:, i * emb_dim:(i + 1) * emb_dim]
                u_final, i_final = unlearned_model.computer(temp_u, temp_i, graph=unlearned_model.Graph[i])
                user_shard_final.append(u_final)
                item_shard_final.append(i_final)

            # [n_users, num_local, emb_dim], [n_items, num_local, emb_dim]
            u_es = torch.stack(user_shard_final, dim=1)
            i_es = torch.stack(item_shard_final, dim=1)

            #          
            if 'trans_W' in unlearned_model.weights and 'trans_B' in unlearned_model.weights:
                u_es = torch.einsum('abc,bcd->abd', u_es, unlearned_model.weights['trans_W']) + unlearned_model.weights['trans_B']
                i_es = torch.einsum('abc,bcd->abd', i_es, unlearned_model.weights['trans_W']) + unlearned_model.weights['trans_B']
            un_user_emb, _ = unlearned_model.attention_based_agg(u_es, 0)
            un_item_emb, _ = unlearned_model.attention_based_agg(i_es, 1)
        else:
            #  LightGCN  
            un_user_emb, un_item_emb = unlearned_model.computer()

    un_pos_orig_ranks = []
    un_pos_unl_ranks = []
    rank_drop_count = 0

    for user_id, un_items in user_unlearn_items.items():
        if not un_items or user_id not in user_interactions:
            continue
        user_items = list(user_interactions[user_id])
        if len(user_items) == 0:
            continue
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



class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.regs_agg = 0
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'Gowalla'
        self.attack = '0.01'
        self.pretrain = 0
        self.n_layers = 1
        self.gcn_layers=1
        self.verbose = 1
        self.data_type = 'full'
        self.save_flag = 1
        self.drop_prob = 0
        self.biased = False
        self.init_std = 1e-3
        self.part_type = 1  # 0: whole data, 1: interaction_based, 2: user_based, 3: random
        self.part_num = 10  # partition number
        self.part_T = 50  # iteration for partition
        self.seed = 1024
        self.n_neg = 1  # number of negative samples per positive
        self.keep_prob=1
        self.A_n_fold=10
        self.A_split=False
        self.dropout=False
        self.pretrain=0

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class early_stoper(object):
    def __init__(self, refer_metric='valid_ndcg', stop_condition=10):
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
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
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

    def re_init(self, stop_condition=None):
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.init_flag = True
        if stop_condition is not None:
            self.stop_condition = stop_condition


def generate_bpr_triplets_from_partition(partition_data, n_users, n_items, n_neg=1):
    users = []
    pos_items = []
    neg_items = []

    if len(partition_data) > 0:
        all_users = [data_point[0] for data_point in partition_data]
        all_items = [data_point[1] for data_point in partition_data]
        min_user_id = min(all_users)
        max_user_id = max(all_users)
        min_item_id = min(all_items)
        max_item_id = max(all_items)

        if min_user_id < 0 or max_user_id >= n_users:
            print(f"ERROR: User IDs out of range! Expected: 0~{n_users-1}, Got: {min_user_id}~{max_user_id}")
            return np.array([]), np.array([]), np.array([])

        if min_item_id < 0 or max_item_id >= n_items:
            print(f"ERROR: Item IDs out of range! Expected: 0~{n_items-1}, Got: {min_item_id}~{max_item_id}")
            return np.array([]), np.array([]), np.array([])

    for data_point in partition_data:
        user = data_point[0]
        pos_item = data_point[1]

        if user < 0 or user >= n_users or pos_item < 0 or pos_item >= n_items:
            print(f"Warning: Data point {data_point} exceeds range. Skipping...")
            print(f"User {user} should be 0~{n_users-1}, Item {pos_item} should be 0~{n_items-1}")
            continue

        for _ in range(n_neg):
            neg_item = random.randint(0, n_items - 1)
            while neg_item == pos_item:
                neg_item = random.randint(0, n_items - 1)

            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)

    if len(users) == 0:
        print("Warning: No valid triplets generated!")
        return np.array([]), np.array([]), np.array([])

    print(f"Generated {len(users)} valid triplets")
    return np.array(users), np.array(pos_items), np.array(neg_items)


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

    config = dict()
    data_generator = Data_for_RecEraser_LightGCN(args.data_path + args.dataset + '/' + args.attack, args.batch_size, args.part_type, args.part_num, args.part_T)
    data_generator.set_train_mode(args.data_type)
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    #       
    print(f"Data range - Users: 0~{data_generator.n_users-1}, Items: 0~{data_generator.n_items-1}")
    for i in range(args.part_num):
        if len(data_generator.C_itr[i]) > 0:
            partition_users = [data_point[0] for data_point in data_generator.C_itr[i]]
            partition_items = [data_point[1] for data_point in data_generator.C_itr[i]]
            max_user_id = max(partition_users)
            max_item_id = max(partition_items)
            print(f"Partition {i}: max_user_id={max_user_id}, max_item_id={max_item_id}")
            if max_user_id >= data_generator.n_users or max_item_id >= data_generator.n_items:
                print(f"ERROR: Partition {i} contains IDs beyond the model's capacity!")
                print(f"Model capacity: users={data_generator.n_users}, items={data_generator.n_items}")
                sys.exit(1)

    total_time = [0] * (args.part_num + 1)

    model = RecEraser_LightGCN(config, args).cuda()
    model.Graph = data_generator.Graph

    #    
    print(f"Model embedding sizes:")
    print(f"User embedding: {model.user_embedding.weight.shape}")
    print(f"Item embedding: {model.item_embedding.weight.shape}")
    print(f"Expected: users={data_generator.n_users}, items={data_generator.n_items}")

    e_stoper = early_stoper(refer_metric='valid_ndcg', stop_condition=10)

    opt_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for _ in range(args.part_num)]
    agg_params = [p for n, p in model.named_parameters() if not (n.startswith('user_embedding') or n.startswith('item_embedding'))]
    if len(agg_params) == 0:
        agg_params = list(model.parameters())
    opt_list.append(torch.optim.Adam(agg_params, lr=args.lr))

    if args.save_flag == 1:
        weights_save_path = './Weights/LightGCN_SISA_BPR/lightgcn_'
        for name_, val_ in config_args.items():
            weights_save_path += name_ + '_' + str(val_) + '_'
        ensureDir(weights_save_path)

    # train local sub-models with BPR
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_total_start_time = time()
    local_train_start_time = time()
    for i in range(args.part_num):
        e_stoper.re_init()
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"
        print("start training %d-th sub-model" % (i))

        partition_data = data_generator.C_itr[i]
        for epoch in range(args.epoch):
            t1 = time()
            loss, bpr_loss, reg_loss = 0., 0., 0.
            model.train()

            users, pos_items, neg_items = generate_bpr_triplets_from_partition(
                partition_data, data_generator.n_users, data_generator.n_items, args.n_neg
            )

            if len(users) == 0:
                print(f"Warning: No valid triplets for partition {i}, epoch {epoch}. Skipping...")
                continue

            n_batches = len(users) // args.batch_size + (1 if len(users) % args.batch_size != 0 else 0)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min((batch_idx + 1) * args.batch_size, len(users))

                batch_users = torch.from_numpy(users[start_idx:end_idx]).cuda().long()
                batch_pos_items = torch.from_numpy(pos_items[start_idx:end_idx]).cuda().long()
                batch_neg_items = torch.from_numpy(neg_items[start_idx:end_idx]).cuda().long()

                model.zero_grad()
                user_emb, _ = model.emb_lookup(batch_users, batch_pos_items, local_id=i)
                _, pos_item_emb = model.emb_lookup(batch_users, batch_pos_items, local_id=i)
                _, neg_item_emb = model.emb_lookup(batch_users, batch_neg_items, local_id=i)
                batch_bpr_loss, batch_reg_loss = model.compute_bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                batch_loss = batch_bpr_loss + batch_reg_loss

                batch_loss.backward()
                opt_list[i].step()

                loss += float(batch_loss.item())
                bpr_loss += float(batch_bpr_loss.item()) if torch.is_tensor(batch_bpr_loss) else float(batch_bpr_loss)
                reg_loss += float(batch_reg_loss.item()) if torch.is_tensor(batch_reg_loss) else float(batch_reg_loss)

            if torch.isnan(torch.tensor(loss)) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            t2 = time()
            k_list = [20]
            valid_metrics = compute_ranking_metrics_sisa(model, data_generator.valid, data_generator, k_list, local_id=i)
            valid_ndcg = valid_metrics['ndcg_at_20']
            one_result = {'valid_ndcg': valid_ndcg}

            t3 = time()
            if args.verbose > 0:
                perf_str = '[local_model %d] epoch %d [%.4fs + %.4fs]: train_loss=[%.4f=%.4f + %.4f], [valid] ndcg@20=[%.4f]' \
                            % \
                           (i, epoch, t2 - t1, t3 - t2, loss, bpr_loss, reg_loss, one_result['valid_ndcg'])
                print(perf_str)

            is_best = e_stoper.update_and_isbest(one_result, epoch)
            if is_best:
                total_time[i] += (t2 - t1)
                user_emb = model.user_embedding.weight[:, i * (model.emb_dim):(i + 1) * model.emb_dim].detach().cpu().numpy()
                item_emb = model.item_embedding.weight[:, i * (model.emb_dim):(i + 1) * model.emb_dim].detach().cpu().numpy()
                with open(weights_save_path_local, 'wb') as f:
                    pickle.dump((user_emb, item_emb), f)

            if e_stoper.is_stop():
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[TIME] Local sub-model training (all partitions) elapsed: {time() - local_train_start_time:.4f}s")
    # aggregate local embeddings
    user_emb = []
    item_emb = []
    for i in range(args.part_num):
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"
        with open(weights_save_path_local, 'rb') as f:
            emb1, emb2 = pickle.load(f)
            user_emb.append(emb1)
            item_emb.append(emb2)
    user_emb = np.concatenate(user_emb, axis=-1)
    item_emb = np.concatenate(item_emb, axis=-1)
    user_emb = torch.from_numpy(user_emb).float().cuda()
    item_emb = torch.from_numpy(item_emb).float().cuda()
    with torch.no_grad():
        model.user_embedding.weight.copy_(user_emb.data)
        model.item_embedding.weight.copy_(item_emb.data)

    print(config_args)

    # train aggregate model with BPR
    e_stoper.re_init()
    agg_train_loader = data_generator.batch_generator()
    for epoch in range(args.epoch):
        t1 = time()
        loss, bpr_loss, reg_loss = 0., 0., 0.
        model.train()
        for batch_data in agg_train_loader:
            users, items, labels = batch_data[:, 0].cuda().long(), batch_data[:, 1].cuda().long(), batch_data[:, 2].cuda().float()
            pos_mask = labels == 1
            if pos_mask.sum() == 0:
                continue

            pos_users = users[pos_mask]
            pos_items = items[pos_mask]
            neg_items = torch.randint(0, data_generator.n_items, (pos_users.shape[0],), device=pos_users.device).long()

            model.zero_grad()
            batch_loss, batch_bpr_loss, batch_reg_loss = model.compute_agg_model_bpr(pos_users, pos_items, neg_items)

            batch_loss.backward()
            opt_list[-1].step()

            loss += float(batch_loss.item())
            bpr_loss += float(batch_bpr_loss.item()) if torch.is_tensor(batch_bpr_loss) else float(batch_bpr_loss)
            reg_loss += float(batch_reg_loss.item()) if torch.is_tensor(batch_reg_loss) else float(batch_reg_loss)

        if torch.isnan(torch.tensor(loss)) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        k_list = [20]
        valid_metrics = compute_ranking_metrics_sisa(model, data_generator.valid, data_generator, k_list)
        valid_ndcg = valid_metrics['ndcg_at_20']
        one_result = {'valid_ndcg': valid_ndcg}

        t3 = time()
        if args.verbose > 0:
            print("epoch: %d, time: %.6f, bpr_loss:%.6f, reg_loss:%.6f, valid ndcg@20:%.6f" %(epoch, t3-t2, bpr_loss, reg_loss, one_result['valid_ndcg']))

        is_best = e_stoper.update_and_isbest(one_result, epoch)
        if is_best:
            total_time[-1] += (t2-t1)
            torch.save(model.state_dict(), weights_save_path + "-m.pth")

        if e_stoper.is_stop():
            print("best epoch: ", e_stoper.best_epoch)
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[TIME] Total model training elapsed: {time() - training_total_start_time:.4f}s")
    # After Unlearning  
    print("***************after unlearning***************")
    k_list = [20, 50, 100]
    try:
        test_metrics_after = compute_ranking_metrics_sisa(model, data_generator.test, data_generator, k_list)
        valid_metrics_after = compute_ranking_metrics_sisa(model, data_generator.valid, data_generator, k_list)
        for k in k_list:
            print(f"Valid Recall@{k}: {valid_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {valid_metrics_after[f'ndcg_at_{k}']:.4f}")
            print(f"Test  Recall@{k}: {test_metrics_after[f'recall_at_{k}']:.4f}, NDCG@{k}: {test_metrics_after[f'ndcg_at_{k}']:.4f}")
    except Exception as e:
        print(f"Warning: After unlearning ranking metrics calculation failed: {e}")

    #   
    print("***************comprehensive ranking metrics***************")
    origin_generator = Data_for_LightGCN(args, path='Data/Process/'+ args.dataset + '/' + args.attack)
    origin_generator.set_train_mode(mode='original')
    original_model = LightGCN(args, dataset=origin_generator).cuda()
    original_save_name = f'./Weights/LightGCN/LightGCN_lr-{0.001}-embed_size-{args.embed_size}-batch_size-2048-data_type-original-dataset-{args.dataset}-attack-{args.attack}-seed-{args.seed}-init_std-0.001-gcn_layers-{args.gcn_layers}-n_neg-1-m.pth'
    original_model.load_state_dict(torch.load(original_save_name))

    # set for compute_comprehensive_ranking_metrics
    comprehensive_ranking_metrics = compute_comprehensive_ranking_metrics_lightgcn(original_model, model, data_generator, top_k=10)
    print("===   Positive Interaction Metrics ===")
    print("  positive interaction   :", comprehensive_ranking_metrics['unlearn_positive_rank_drop_ratio'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_original'])
    print("    positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_avg_rank_unlearned'])
    print("  positive interaction  :", comprehensive_ranking_metrics['unlearn_positive_rank_change'])

    #  (+)    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[TIME] End-to-end process (train + evaluation) elapsed: {time() - training_total_start_time:.4f}s")





if __name__ == '__main__':
    config = {
        'lr': 1e-3,
        'embed_size': 48,
        'batch_size': 2048,
        'data_type': 'retraining',
        'init_std': 1e-4,
        'dataset': 'Gowalla',
        'attack': '0.01',
        'seed': 1024,
        'part_type': 3,  # 0: whole data, 1: interaction_based, 2: user_based, 3: random
        'n_neg': 1,
        'gcn_layers': 1,
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
    
    