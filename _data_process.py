import pandas as pd
import numpy as np
import os
import argparse

# attack = 0.001 # 0.02
# thre_ = 4
# load_path = './Data/Original/Amazon_Electronics.csv'
# save_path = './Data/Process/Amazon/' + str(attack) + '/'

# attack = 0.001 # 0.02
# thre_ = 5
# load_path = './Data/Original/BookCrossing.csv'
# save_path = './Data/Process/BookCrossing/' + str(attack) + '/'

# Yelp, Gowalla, Amazon_Book   
# attack = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess datasets (Amazon_Book, Gowalla, Yelp)')
    parser.add_argument('--dataset', type=str, choices=['Amazon_Book', 'Gowalla', 'Yelp'], required=True,
                        help='Target dataset')
    parser.add_argument('--attack', type=float, default=0.01, help='Attack ratio (0-1) for unlearning split')
    parser.add_argument('--k', type=int, default=5, help='k-core value')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Train ratio (0-1)')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Validation ratio (0-1)')
    parser.add_argument('--data_root', type=str, default='./Data', help='Root directory for Original data')
    parser.add_argument('--process_root', type=str, default='./Data/Process', help='Root directory to save processed data')
    return parser.parse_args()


# thre_ = 6  # Yelp
# load_path = './Data/Original/Yelp2018.txt'
# save_path = './Data/Process/Yelp/' + str(attack) + '/'

# thre_ = 7  # Gowalla
# load_path = './Data/Original/Gowalla.txt'
# save_path = './Data/Process/Gowalla/' + str(attack) + '/'

# thre_ = 8  # Amazon_Book
# load_path = './Data/Original/Amazon_Book.txt'
# save_path = './Data/Process/Amazon_Book/' + str(attack) + '/'

if __name__ == '__main__':

    args = parse_args()
    dataset_to_filename = {
        'Yelp': 'Yelp2018.txt',
        'Gowalla': 'Gowalla.txt',
        'Amazon_Book': 'Amazon_Book.txt',
    }
    load_path = os.path.join(args.data_root, 'Original', dataset_to_filename[args.dataset])
    save_path = os.path.join(args.process_root, args.dataset, str(args.attack), '')

    print(args.dataset)
    data = pd.read_csv(load_path, delimiter=' ', header=None, names=['user', 'item', 'label'])

    # thre_ 6, 7, 8      
    # (     )

    print("user num:", data['user'].unique().shape[0])
    print("item num:", data['item'].unique().shape[0])
    print("intersection num:", data.shape[0])
    print("data sparse:", data.shape[0] / data['user'].unique().shape[0] / data['item'].unique().shape[0])
    print("label type:", sorted(data['label'].unique()))

    def k_core_filtering(select_data, k=5):
        itr = 0
        while True:
            pre_shape = select_data.shape[0]

            user_info = select_data.groupby('user').agg({'label': ['count', 'mean']})
            s_u = user_info[user_info[('label', 'count')] >= k].index
            select_data = select_data[select_data['user'].isin(s_u)]

            item_info = select_data.groupby('item').agg({'label': ['count', 'mean']})
            s_i = item_info[item_info[('label', 'count')] >= k].index
            select_data = select_data[select_data['item'].isin(s_i)]

            aft_shape = select_data.shape[0]
            print("itr:", itr, 'pre-shape:', pre_shape, 'aft_shape:', aft_shape)
            if pre_shape == aft_shape:
                break
            itr += 1
        return select_data

    select_data = k_core_filtering(data, k=args.k)
    print("user num:", select_data['user'].unique().shape[0])
    print("item num:", select_data['item'].unique().shape[0])
    print("intersection num:", select_data.shape[0])
    print("data sparse:", select_data.shape[0] / select_data['user'].unique().shape[0] / select_data['item'].unique().shape[0])

    idx = np.arange(select_data.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(idx)
    n1 = int(select_data.shape[0] * args.train_ratio)
    n2 = int(select_data.shape[0] * (args.train_ratio + args.valid_ratio))
    idx1 = idx[:n1]
    idx2 = idx[n1:n2]
    idx3 = idx[n2:]
    train, valid, test = select_data.iloc[idx1], select_data.iloc[idx2], select_data.iloc[idx3]

    print("user num:", train['user'].unique().shape[0])
    print("item num:", train['item'].unique().shape[0])
    print("intersection num:", train.shape[0])

    user_map = dict(zip(train['user'].unique(), np.arange(train['user'].unique().shape[0])))  
    item_map = dict(zip(train['item'].unique(), np.arange(train['item'].unique().shape[0])))  

    def enconding_f(pd_data, user_map, item_map):  
        pd_data = pd_data.copy()
        pd_data['user'] = pd_data['user'].map(user_map)
        pd_data['item'] = pd_data['item'].map(item_map)
        # Yelp, Gowalla, Amazon_Book   rating 1  1 
        pd_data['label'] = 1
        return pd_data
    
    train_user = train['user'].unique()
    train_item = train['item'].unique()
    valid_ = valid[valid['user'].isin(train_user)]
    valid_ = valid_[valid_['item'].isin(train_item)]
    test_ = test[test['user'].isin(train_user)]
    test_ = test_[test_['item'].isin(train_item)]

    nn = int(train.shape[0] * (1 - args.attack))
    idx1 = np.arange(nn)
    idx2 = np.arange(nn, train.shape[0])
    train_normal, train_unlearn = train.iloc[idx1].copy(), train.iloc[idx2].copy()

    train_normal_save = enconding_f(train_normal, user_map, item_map)
    train_unlearn_save = enconding_f(train_unlearn, user_map, item_map)
    train_original_save = enconding_f(train, user_map, item_map)
    valid_save = enconding_f(valid_, user_map, item_map)
    test_save = enconding_f(test_, user_map, item_map)

    train_unlearn_original = train_unlearn_save.copy(deep=True)
    train_unlearn_save['label'] = 1 - train_unlearn_save['label']

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    train_original_save.to_csv(save_path + 'train_original.csv', index=None)
    train_normal_save.to_csv(save_path + 'train_normal.csv', index=None)
    train_unlearn_save.to_csv(save_path + 'train_random.csv', index=None)
    train_unlearn_original.to_csv(save_path + 'train_unlearn_original.csv', index=None)
    test_save.to_csv(save_path + 'test.csv', index=None)
    valid_save.to_csv(save_path + 'valid.csv', index=None)