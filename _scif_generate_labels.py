import pandas as pd
import numpy as np
import argparse
import os

# path = './Data/Process/BookCrossing/0.02/'
# path = './Data/Process/Amazon/0.02/'
# path = './Data/Process/BookCrossing/0.01/'
# path = './Data/Process/Amazon/0.01/'
# path = './Data/Process/Yelp/0.01/'
# path = './Data/Process/Gowalla/0.01/'
# path = './Data/Process/Amazon_Book/0.01/'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate soft labels based on train_normal and train_random.')
    parser.add_argument('--dataset', type=str, choices=['Amazon_Book', 'Gowalla', 'Yelp'], required=True,
                        help='Target dataset')
    parser.add_argument('--attack', type=str, default='0.01', help='Attack ratio directory name, e.g., 0.01')
    parser.add_argument('--process_root', type=str, default='./Data/Process', help='Root directory of processed data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    path = os.path.join(args.process_root, args.dataset, args.attack)

    # 读取train_normal和train_random的CSV文件
    train_normal_df = pd.read_csv(os.path.join(path, 'train_normal.csv'))
    train_random_df = pd.read_csv(os.path.join(path, 'train_random.csv'))

    # 创建一个空的numpy数组来存储替换后的label
    new_labels = np.zeros(len(train_random_df))

    # 遍历train_random中的每一行
    for index, row in train_random_df.iterrows():
        u = row['user']
        i = row['item']
        user_records = train_normal_df[train_normal_df['user'] == u]
        
        if len(user_records) > 0:
            # 计算指定user的平均label
            avg_label = user_records['label'].mean()
        else:
            item_records = train_normal_df[train_normal_df['item'] == i]
            # 计算指定item的平均label
            avg_label = item_records['label'].mean()
        
        # 将新的label存储到numpy数组中
        new_labels[index] = avg_label

    # 将新的labels保存为npy格式
    np.save(os.path.join(path, 'avg_labels.npy'), new_labels)