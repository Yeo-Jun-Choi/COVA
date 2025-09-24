import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


def get_random_negative_samples(pos_items: np.ndarray, n_items: int, n_neg: int = 1) -> np.ndarray:
    """
     :   random negative sampling ( )
    
    Args:
        pos_items: positive  ID (batch_size,)
        n_items:   
        n_neg: negative sample  (: 1)
    
    Returns:
        neg_items: negative  ID (batch_size, n_neg)
    """
    batch_size = len(pos_items)
    neg_items = np.zeros((batch_size, n_neg), dtype=np.int64)
    
    #   sampling (positive   )
    for i in range(batch_size):
        neg_items[i] = np.random.choice(n_items, size=n_neg, replace=False)
    
    return neg_items


def get_negative_samples_in_group(pos_items: np.ndarray, user_groups: np.ndarray, 
                                 group_item_dict: Dict[int, List[int]], n_neg: int = 1) -> np.ndarray:
    """
    Partition :  group  negative sampling
    
    Args:
        pos_items: positive  ID (batch_size,)
        user_groups:   group ID (batch_size,)
        group_item_dict: group   {group_id: [item_ids]}
        n_neg: negative sample  (: 1)
    
    Returns:
        neg_items: negative  ID (batch_size, n_neg)
    """
    batch_size = len(pos_items)
    neg_items = np.zeros((batch_size, n_neg), dtype=np.int64)
    
    for i in range(batch_size):
        group_id = user_groups[i]
        pos_item = pos_items[i]
        
        #  group    positive  
        group_items = group_item_dict.get(group_id, [])
        available_items = list(set(group_items) - {pos_item})
        
        if len(available_items) >= n_neg:
            neg_items[i] = np.random.choice(available_items, size=n_neg, replace=False)
        else:
            # group    ,  sampling
            all_items = list(set(range(max(group_items) + 1)) - {pos_item})
            neg_items[i] = np.random.choice(all_items, size=n_neg, replace=False)
    
    return neg_items


def create_group_item_dict(partition_data: np.ndarray) -> Dict[int, List[int]]:
    """
    Partition  group   
    
    Args:
        partition_data: partition    (user_id, item_id, group_id, ...)
    
    Returns:
        group_item_dict: {group_id: [item_ids]}
    """
    group_item_dict = {}
    
    for row in partition_data:
        group_id = row[2]  # group_id 3  
        item_id = row[1]   # item_id 2  
        
        if group_id not in group_item_dict:
            group_item_dict[group_id] = []
        
        if item_id not in group_item_dict[group_id]:
            group_item_dict[group_id].append(item_id)
    
    return group_item_dict


def sample_triplets_for_bpr(pos_data: np.ndarray, n_items: int, 
                           user_groups: Optional[np.ndarray] = None,
                           group_item_dict: Optional[Dict[int, List[int]]] = None,
                           n_neg: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BPR   triplet sampling
    
    Args:
        pos_data: positive  (user_id, item_id, ...)
        n_items:   
        user_groups:  group  (partition )
        group_item_dict: group   (partition )
        n_neg: negative sample 
    
    Returns:
        users:  ID
        pos_items: positive  ID
        neg_items: negative  ID
    """
    users = pos_data[:, 0].astype(np.int64)
    pos_items = pos_data[:, 1].astype(np.int64)
    
    if user_groups is not None and group_item_dict is not None:
        # Partition :  group  sampling
        neg_items = get_negative_samples_in_group(pos_items, user_groups, group_item_dict, n_neg)
    else:
        #  :  random sampling
        neg_items = get_random_negative_samples(pos_items, n_items, n_neg)
    
    return users, pos_items, neg_items 