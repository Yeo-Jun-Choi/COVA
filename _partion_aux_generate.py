import torch
import numpy as np
import pickle
from Model.MF import MF
from utility.load_data import *
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Generate MF pretrain embeddings from saved weights.')
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
    parser.add_argument('--n_neg', type=int, default=1, help='Number of negatives used (for weight name)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_path = os.path.join(args.process_root, args.dataset, args.attack)
    load_path = os.path.join(
        args.weights_root,
        'MF',
        f"MF_lr-{args.lr}-embed_size-{args.embed_size}-batch_size-{args.batch_size}-data_type-original-dataset-{args.dataset}-attack-{args.attack}-seed-{args.seed}-init_std-{args.init_std}-n_neg-{args.n_neg}-m.pth"
    )
    embed_size = args.embed_size

    data_generator = Data_for_MF(data_path=data_path, batch_size=args.batch_size)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    class MyObject:
        def __init__(self):
            self.lr = args.lr
            self.embed_size = embed_size
            self.batch_size = args.batch_size
            self.regs = 0
            self.init_std = args.init_std
            self.n_neg = args.n_neg
    args_obj = MyObject()

    model = MF(data_config=config, args=args_obj).cuda()
    model.load_state_dict(torch.load(load_path))

    user_pretrain = model.user_embeddings.weight.cpu().detach().numpy()
    item_pretrain = model.item_embeddings.weight.cpu().detach().numpy()
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, 'user_pretrain.pk'),'wb') as f:
        pickle.dump(user_pretrain,f)
    with open(os.path.join(data_path, 'item_pretrain.pk'),'wb') as f:
        pickle.dump(item_pretrain,f)
