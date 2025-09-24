from distutils.command.config import config
import os
import torch
import numpy as np
from utility.load_data import *
import pandas as pd
import sys
from time import time
from sklearn.metrics import roc_auc_score
from utility.compute import compute_ranking_metrics
import random
from Model.MF import MF
from utility.compute import *
from utility.negative_sampling import sample_triplets_for_bpr


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.embed_size = 48
        self.regs = 0
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'Amazon_Book'
        self.attack = '0.01'
        self.data_type = 'full'
        self.seed = 1024
        self.init_std = 1e-4
        self.n_neg = 1  # negative sample 

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class early_stoper(object):
    def __init__(self, refer_metric='valid_auc', stop_condition=10):
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

    save_name = 'MF_'
    for name_str, name_val in config_args.items():
        save_name += name_str + '-' + str(name_val) + '-'

    data_generator = Data_for_MF(data_path=args.data_path + args.dataset + '/' + args.attack, batch_size=args.batch_size)
    data_generator.set_train_mode(args.data_type)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    model = MF(data_config=config, args=args).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=model.lr)

    best_epoch = 0
    best_valid_recall = 0
    best_test_recall = 0
    e_stoper = early_stoper(refer_metric='valid_recall_20', stop_condition=10)
    print("Start training")
    for epoch in range(args.epoch):
        t1 = time()
        loss, bpr_loss, reg_loss = 0., 0., 0.
        
        # Positive   BPR 
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
            
            batch_bpr_loss, batch_reg_loss, batch_loss = model.train_one_batch_bpr(
                batch_users, batch_pos_items, batch_neg_items, opt
            )
            
            loss += batch_loss
            bpr_loss += batch_bpr_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        # Ranking metrics 
        k_list = [20, 50, 100]
        valid_metrics = compute_ranking_metrics(model, data_generator.valid, data_generator, k_list)
        test_metrics = compute_ranking_metrics(model, data_generator.test, data_generator, k_list)

        t3 = time()
        perf_str = "epoch: %d, time: %.6f, bpr_loss:%.6f, reg_loss:%.6f" %(epoch, t3-t2, bpr_loss, reg_loss)
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
            torch.save(model.state_dict(), './Weights/MF/' + save_name + "m.pth")
            print("saving the best model")

        if e_stoper.is_stop():
            print("save path for best model:", './Weights/MF/' + save_name + "m.pth")
            break

    final_perf = 'best_epoch = {}, best_valid_recall@20 = {}, best_test_recall@20 = {}'.format(best_epoch, best_valid_recall, best_test_recall)
    print(final_perf)


if __name__ == '__main__':
    config = {
        'lr': 1e-3,  # [1e-2, 1e-3, 1e-4]
        'embed_size': 48,  # [32, 48, 64]
        'batch_size': 2048,
        'data_type': 'original',
        'dataset': 'Amazon_Book',  #[Yelp, Gowalla, Amazon_Book]
        'attack':'0.01',  # [0.02, 0.01]
        'seed': 1024,
        'init_std': 1e-3,  # [1e-2, 1e-3, 1e-4]
        'n_neg': 1  # negative sample 
    }
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str)
    _args = parser.parse_args()
    if _args.attack is not None:
        config['attack'] = _args.attack
    main(config) 