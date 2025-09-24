import torch
import numpy as np
import pickle
# from Model.MF import MF
from Model.Lightgcn import LightGCN
from utility.load_data import *
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Generate LightGCN pretrain embeddings from saved weights.')
    parser.add_argument('--dataset', type=str, choices=['Amazon_Book', 'Gowalla', 'Yelp'], required=True,
                        help='Target dataset')
    parser.add_argument('--attack', type=str, default='0.01', help='Attack ratio directory name, e.g., 0.01')
    parser.add_argument('--process_root', type=str, default='./Data/Process', help='Root directory of processed data')
    parser.add_argument('--weights_root', type=str, default='./Weights', help='Root directory of saved model weights')
    parser.add_argument('--embed_size', type=int, default=48, help='Embedding size used in the trained model')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size used in the trained model')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed used in the trained model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate used in the trained model (for weight name)')
    parser.add_argument('--init_std', type=float, default=0.001, help='Init std used in the trained model (for weight name)')
    parser.add_argument('--gcn_layers', type=int, default=1)
    parser.add_argument('--keep_prob', type=float, default=1)
    parser.add_argument('--A_n_fold', type=int, default=10)
    parser.add_argument('--A_split', action='store_true', default=False)
    parser.add_argument('--dropout', action='store_true', default=False)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--n_neg', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_path = os.path.join(args.process_root, args.dataset, args.attack)
    load_path_new = os.path.join(
        args.weights_root,
        'LightGCN',
        f"LightGCN_lr-{args.lr}-embed_size-{args.embed_size}-batch_size-{args.batch_size}-data_type-original-dataset-{args.dataset}-attack-{args.attack}-seed-{args.seed}-init_std-{args.init_std}-n_neg-{args.n_neg}-gcn_layers-{args.gcn_layers}-m.pth"
    )
    load_path_old = os.path.join(
        args.weights_root,
        'LightGCN',
        f"LightGCN_lr-{args.lr}-embed_size-{args.embed_size}-batch_size-{args.batch_size}-data_type-original-dataset-{args.dataset}-attack-{args.attack}-seed-{args.seed}-init_std-{args.init_std}-n_neg-{args.n_neg}-m.pth"
    )
    load_path = load_path_new if os.path.exists(load_path_new) else load_path_old

    class Args:
        def __init__(self):
            self.lr = args.lr
            self.embed_size = args.embed_size
            self.batch_size = args.batch_size
            self.regs = 0
            self.init_std = args.init_std
            self.gcn_layers = args.gcn_layers
            self.keep_prob = args.keep_prob
            self.A_n_fold = args.A_n_fold
            self.A_split = args.A_split
            self.dropout = args.dropout
            self.pretrain = args.pretrain
            self.dataset = args.dataset
            self.n_neg = args.n_neg
    model_args = Args()

    # LightGCN   
    data_generator = Data_for_LightGCN(model_args, path=data_path)

    # LightGCN     
    model = LightGCN(model_args, dataset=data_generator).cuda()
    model.load_state_dict(torch.load(load_path))

    #   (train_original )
    data_generator.set_train_mode('original')
    # set_train_mode   Graph  
    model.Graph = data_generator.Graph

    # MP   
    all_user_emb, all_item_emb = model.computer()
    user_pretrain = all_user_emb.cpu().detach().numpy()
    item_pretrain = all_item_emb.cpu().detach().numpy()
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, 'LightGCN_user_pretrain.pk'),'wb') as f:
        pickle.dump(user_pretrain,f)
    with open(os.path.join(data_path, 'LightGCN_item_pretrain.pk'),'wb') as f:
        pickle.dump(item_pretrain,f)
