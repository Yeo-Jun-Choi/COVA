import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score


def compute_neighbor(data_generator, k_hop=0):
    assert k_hop == 0
    train_data = data_generator.train.values.copy()
    matrix_size = data_generator.n_users + data_generator.n_items
    train_data[:,1] += data_generator.n_users
    train_data[:,-1] = np.ones_like(train_data[:,-1])

    train_data2 = np.ones_like(train_data)
    train_data2[:,0] = train_data[:,1]
    train_data2[:,1] = train_data[:,0]

    paddding = np.concatenate([np.arange(matrix_size).reshape(-1,1), np.arange(matrix_size).reshape(-1,1), np.ones(matrix_size).reshape(-1,1)],axis=-1)
    data = np.concatenate([train_data, train_data2, paddding],axis=0).astype(int)
    train_matrix = sp.csc_matrix((data[:,-1],(data[:,0],data[:,1])),shape=(matrix_size,matrix_size))
    
    neighbor_set = list()
    init_users = data_generator.train_random['user'].values.reshape(-1)
    neighbor_set.extend(np.unique(init_users))
    init_items = data_generator.train_random['item'].values.reshape(-1) + data_generator.n_users
    neighbor_set.extend(np.unique(init_items))
    # print("neighbor_set size:", len(neighbor_set))

    neighbor_set = np.array(neighbor_set)
    return neighbor_set[np.where(neighbor_set<data_generator.n_users)], neighbor_set[np.where(neighbor_set>=data_generator.n_users)] - data_generator.n_users


def get_eval_mask(data_generator):

    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values

    nei_users, nei_items = compute_neighbor(data_generator)
    nei_users = torch.from_numpy(nei_users).cuda().long()
    nei_items = torch.from_numpy(nei_items).cuda().long()

    # mask or
    mask_1 = np.zeros(valid_data.shape[0])
    for ii in range(valid_data.shape[0]):
        if valid_data[ii,0] in nei_users or valid_data[ii,1] in nei_items:
            mask_1[ii] = 1
    mask_1 = np.where(mask_1>0)[0] 

    mask_2 = np.zeros(test_data.shape[0])
    for ii in range(test_data.shape[0]):
        if test_data[ii,0] in nei_users or test_data[ii,1] in nei_items:
            mask_2[ii] = 1
    mask_2 = np.where(mask_2>0)[0] 
    
    # mask and
    mask_3 = np.zeros(valid_data.shape[0])
    for ii in range(valid_data.shape[0]):
        if valid_data[ii,0] in nei_users and valid_data[ii,1] in nei_items:
            mask_3[ii] = 1
    mask_3 = np.where(mask_3>0)[0] 

    mask_4 = np.zeros(test_data.shape[0])
    for ii in range(test_data.shape[0]):
        if test_data[ii,0] in nei_users and test_data[ii,1] in nei_items:
            mask_4[ii] = 1
    mask_4 = np.where(mask_4>0)[0] 

    return (mask_1, mask_2, mask_3, mask_4)


def get_eval_result(data_generator, model, mask):
    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values

    nei_users, nei_items = compute_neighbor(data_generator)
    nei_users = torch.from_numpy(nei_users).cuda().long()
    nei_items = torch.from_numpy(nei_items).cuda().long()

    mask_1, mask_2, mask_3, mask_4 = mask[0], mask[1], mask[2], mask[3]

    with torch.no_grad():
        valid_predictions =  model.predict(valid_data[:,0], valid_data[:,1])
        test_predictions =  model.predict(test_data[:,0], test_data[:,1])

    valid_auc = roc_auc_score(valid_data[:,-1],valid_predictions)
    valid_auc_or = roc_auc_score(valid_data[:,-1][mask_1], valid_predictions[mask_1])
    valid_auc_and = roc_auc_score(valid_data[:,-1][mask_3], valid_predictions[mask_3])

    test_auc = roc_auc_score(test_data[:,-1],test_predictions)
    test_auc_or = roc_auc_score(test_data[:,-1][mask_2], test_predictions[mask_2])
    test_auc_and = roc_auc_score(test_data[:,-1][mask_4], test_predictions[mask_4])

    return valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and



def compute_item_ranking_metrics(original_model, unlearned_model, data_generator, top_k=10):
    """
          interaction   
    
    Args:
        original_model:  
        unlearned_model:  
        data_generator:  
        top_k:  k  
    
    Returns:
        dict:   
    """
    #   interaction (train_random )
    unlearn_interactions = data_generator.train_random[['user', 'item']].values
    
    ranking_metrics = {
        'rank_drop_ratio': 0.0,  #   
        'unlearn_interaction_avg_rank_original': 0.0,  #    interaction  
        'unlearn_interaction_avg_rank_unlearned': 0.0,  #    interaction  
        'rank_drop_user_avg_degree': 0.0,  #   interaction user degree 
        'rank_drop_item_avg_degree': 0.0,  #   interaction item degree 
    }
    
    rank_drops_count = 0
    total_unlearn_interactions = 0
    
    original_ranks = []
    unlearned_ranks = []
    
    #     
    user_interactions = {}
    for user_id, item_id in data_generator.train[['user', 'item']].values:
        if user_id not in user_interactions:
            user_interactions[user_id] = set()
        user_interactions[user_id].add(item_id)
    
    #     
    item_interactions = {}
    for user_id, item_id in data_generator.train[['user', 'item']].values:
        if item_id not in item_interactions:
            item_interactions[item_id] = set()
        item_interactions[item_id].add(user_id)
    
    #   interaction user, item degree  
    rank_drop_user_degrees = []
    rank_drop_item_degrees = []
    
    #   interaction   
    for user_id, item_id in unlearn_interactions:
        if user_id not in user_interactions:
            continue

        #    
        user_items = list(user_interactions[user_id])

        #       
        user_scores_original = original_model.batch_rating([user_id], user_items).flatten()
        user_scores_unlearned = unlearned_model.batch_rating([user_id], user_items).flatten()

        #   (   )
        original_ranks_user = np.argsort(np.argsort(-user_scores_original)) + 1
        unlearned_ranks_user = np.argsort(np.argsort(-user_scores_unlearned)) + 1

        #    
        item_idx = user_items.index(item_id)
        
        #   
        original_rank = original_ranks_user[item_idx] / len(user_items)
        unlearned_rank = unlearned_ranks_user[item_idx] / len(user_items)
        
        original_ranks.append(original_rank)
        unlearned_ranks.append(unlearned_rank)
    
        rank_drop = original_rank - unlearned_rank
        total_unlearn_interactions += 1
        
        if rank_drop < 0:
            rank_drops_count += 1
            #  user item degree 
            user_degree = len(user_interactions[user_id])
            item_degree = len(item_interactions.get(item_id, set()))

            rank_drop_user_degrees.append(user_degree)
            rank_drop_item_degrees.append(item_degree)

    if total_unlearn_interactions > 0:
        ranking_metrics['rank_drop_ratio'] = rank_drops_count / total_unlearn_interactions
        ranking_metrics['unlearn_interaction_avg_rank_original'] = np.mean(original_ranks)
        ranking_metrics['unlearn_interaction_avg_rank_unlearned'] = np.mean(unlearned_ranks)
        
        #   interaction user, item degree  
        if len(rank_drop_user_degrees) > 0:
            ranking_metrics['rank_drop_user_avg_degree'] = np.mean(rank_drop_user_degrees)
            ranking_metrics['rank_drop_item_avg_degree'] = np.mean(rank_drop_item_degrees)
    
    return ranking_metrics


def compute_comprehensive_ranking_metrics(original_model, unlearned_model, data_generator, top_k=10):
    """
    Unlearning       ( )

    Args:
        original_model:  
        unlearned_model:  
        data_generator:  
        top_k:  k  
    
    Returns:
        dict:    
    """
    #   interaction (train_random )
    unlearn_interactions = data_generator.train_random[['user', 'item', 'label']].values

    ranking_metrics = {
        'unlearn_positive_rank_drop_ratio': 0.0,
        'unlearn_positive_avg_rank_original': 0.0,
        'unlearn_positive_avg_rank_unlearned': 0.0,
        'unlearn_positive_rank_change': 0.0,
    }
    
    if len(unlearn_interactions) == 0:
        return ranking_metrics
    
    #      (  )
    user_interactions = {}
    for user_id, item_id in data_generator.train[['user', 'item']].values:
        if user_id not in user_interactions:
            user_interactions[user_id] = set()
        user_interactions[user_id].add(item_id)
    
    #  unlearn interactions 
    user_unlearn_items = {}
    for user_id, item_id, label in unlearn_interactions:
        if user_id not in user_interactions:
            continue
        if user_id not in user_unlearn_items:
            user_unlearn_items[user_id] = []
        user_unlearn_items[user_id].append(item_id)
    
    #      
    unlearn_positive_original_ranks = []
    unlearn_positive_unlearned_ranks = []
    unlearn_positive_rank_drops_count = 0
    
    #     
    for user_id, unlearn_items in user_unlearn_items.items():
        if not unlearn_items:
            continue

        user_items = list(user_interactions[user_id])
        
        #       
        user_scores_original = original_model.batch_rating([user_id], user_items).flatten()
        user_scores_unlearned = unlearned_model.batch_rating([user_id], user_items).flatten()
        
        #   ( )
        original_ranks_user = np.argsort(np.argsort(-user_scores_original)) + 1
        unlearned_ranks_user = np.argsort(np.argsort(-user_scores_unlearned)) + 1
        
        #  ID   (  )
        item_to_idx = {item: idx for idx, item in enumerate(user_items)}
        
        #  unlearn    
        for item_id in unlearn_items:
            if item_id not in item_to_idx:
                continue
                
            item_idx = item_to_idx[item_id]
            original_rank = original_ranks_user[item_idx]
            unlearned_rank = unlearned_ranks_user[item_idx]
            
            unlearn_positive_original_ranks.append(original_rank)
            unlearn_positive_unlearned_ranks.append(unlearned_rank)
            
            #   
            if original_rank - unlearned_rank < 0:
                unlearn_positive_rank_drops_count += 1

    #  
    if unlearn_positive_original_ranks:
        ranking_metrics['unlearn_positive_rank_drop_ratio'] = unlearn_positive_rank_drops_count / len(unlearn_positive_original_ranks)
        ranking_metrics['unlearn_positive_avg_rank_original'] = np.mean(unlearn_positive_original_ranks)
        ranking_metrics['unlearn_positive_avg_rank_unlearned'] = np.mean(unlearn_positive_unlearned_ranks)
        ranking_metrics['unlearn_positive_rank_change'] = np.mean(unlearn_positive_original_ranks) - np.mean(unlearn_positive_unlearned_ranks)
    
    #   
    ranking_metrics['overall_effectiveness'] = ranking_metrics['unlearn_positive_rank_drop_ratio']
    
    return ranking_metrics


def compute_retraining_based_metrics(retraining_model, unlearning_model, data_generator, k_list=[20, 50, 100]):
    """
    Retraining  ground truth  Unlearning  Recall@K, NDCG@K  
    Retraining       top-k 
    
    Args:
        retraining_model: Retraining  (ground truth)
        unlearning_model: Unlearning  ( )
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
    
    #        
    user_train_items = {}
    for user_id, item_id in data_generator.train_normal[['user', 'item']].values:
        if user_id not in user_train_items:
            user_train_items[user_id] = set()
        user_train_items[user_id].add(item_id)
    
    #    
    for user_id in range(data_generator.n_users):
        #     
        all_items = np.arange(data_generator.n_items)
        retraining_scores = retraining_model.batch_rating([user_id], all_items).flatten()
        unlearning_scores = unlearning_model.batch_rating([user_id], all_items).flatten()
        
        #      (   )
        train_items = user_train_items.get(user_id, set())
        mask = np.ones(data_generator.n_items, dtype=bool)
        for item_id in train_items:
            mask[item_id] = False
        
        #  
        masked_retraining_scores = retraining_scores.copy()
        masked_unlearning_scores = unlearning_scores.copy()
        masked_retraining_scores[~mask] = -np.inf
        masked_unlearning_scores[~mask] = -np.inf
        
        # Top-max_k  (  )
        retraining_top_max_k_indices = np.argsort(-masked_retraining_scores)[:max_k]
        retraining_top_max_k_items = [all_items[i] for i in retraining_top_max_k_indices]
        
        unlearning_top_max_k_indices = np.argsort(-masked_unlearning_scores)[:max_k]
        unlearning_top_max_k_items = [all_items[i] for i in unlearning_top_max_k_indices]
        
        #  K  
        for k in k_list:
            # Top-k  
            retraining_top_k_items = retraining_top_max_k_items[:k]
            unlearning_top_k_items = unlearning_top_max_k_items[:k]
            
            # Recall@K 
            relevant_items = set(retraining_top_k_items)
            recommended_items = set(unlearning_top_k_items)
            recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items) if len(relevant_items) > 0 else 0
            metrics[f'recall_at_{k}'].append(recall)
            
            # NDCG@K 
            dcg = 0
            idcg = 0
            
            # DCG 
            for i, item_id in enumerate(unlearning_top_k_items):
                if item_id in relevant_items:
                    dcg += 1 / np.log2(i + 2)  # log2(i+2) because i starts from 0
            
            # IDCG  (ideal ranking)
            for i in range(min(k, len(relevant_items))):
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg_at_{k}'].append(ndcg)
    
    #  
    for k in k_list:
        metrics[f'recall_at_{k}'] = np.mean(metrics[f'recall_at_{k}'])
        metrics[f'ndcg_at_{k}'] = np.mean(metrics[f'ndcg_at_{k}'])
    
    return metrics
    

def compute_ranking_metrics(model, dataset, data_generator, k_list=[20, 50, 100]):
    """
       NDCG@K, Recall@K  
    
    Args:
        model:  
        dataset:   (valid  test)
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
    for user_id, item_id, label in dataset[['user', 'item', 'label']].values:
        if label == 1:  # positive interaction
            if user_id not in positive_interactions:
                positive_interactions[user_id] = set()
            positive_interactions[user_id].add(item_id)
    
    #    
    for user_id in range(data_generator.n_users):
        if user_id not in positive_interactions or len(positive_interactions[user_id]) == 0:
            continue
            
        #     
        all_items = np.arange(data_generator.n_items)
        user_scores = model.batch_rating([user_id], all_items).flatten()
        
        # Top-max_k 
        top_max_k_indices = np.argsort(-user_scores)[:max_k]
        top_max_k_items = [all_items[i] for i in top_max_k_indices]
        
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


def compute_positive_metrics(data_generator, model, k_list=[20, 50, 100]):
    """
    Positive   NDCG@K, Recall@K  
    
    Args:
        data_generator:  
        model:  
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
    for user_id, item_id, label in data_generator.test[['user', 'item', 'label']].values:
        if label == 1:  # positive interaction
            if user_id not in positive_interactions:
                positive_interactions[user_id] = set()
            positive_interactions[user_id].add(item_id)
    
    #    
    for user_id in range(data_generator.n_users):
        if user_id not in positive_interactions or len(positive_interactions[user_id]) == 0:
            continue
            
        #     
        all_items = np.arange(data_generator.n_items)
        user_scores = model.batch_rating([user_id], all_items).flatten()
        
        # Top-max_k 
        top_max_k_indices = np.argsort(-user_scores)[:max_k]
        top_max_k_items = [all_items[i] for i in top_max_k_indices]
        
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


def compute_ranking_metrics_sisa(model, dataset, data_generator, k_list=[20, 50, 100], local_id=None):
    """
    SISA   NDCG@K, Recall@K  
    
    Args:
        model:  SISA 
        dataset:   (valid  test)
        data_generator:  
        k_list:  K  
        local_id: local partition ID (None aggregate model )
    
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
    for user_id, item_id, label in dataset[['user', 'item', 'label']].values:
        if label == 1:  # positive interaction
            if user_id not in positive_interactions:
                positive_interactions[user_id] = set()
            positive_interactions[user_id].add(item_id)
    
    #    
    for user_id in range(data_generator.n_users):
        if user_id not in positive_interactions or len(positive_interactions[user_id]) == 0:
            continue
            
        #     
        all_items = np.arange(data_generator.n_items)
        
        try:
            if local_id is not None:
                # Local partition  
                user_scores = model.batch_rating_local([user_id], all_items, local_id).flatten()
            else:
                # Aggregate  
                user_scores = model.batch_rating([user_id], all_items).flatten()
        except Exception as e:
            print(f"Warning: Score calculation failed for user {user_id}: {e}")
            break
            continue
        
        # Top-max_k 
        top_max_k_indices = np.argsort(-user_scores)[:max_k]
        top_max_k_items = [all_items[i] for i in top_max_k_indices]
        
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


def compute_comprehensive_ranking_metrics_sisa(original_model, unlearned_model, data_generator, top_k=10):
    """
    SISA   Unlearning      
    
    Args:
        original_model:  SISA 
        unlearned_model:  SISA 
        data_generator:  
        top_k:  k  
    
    Returns:
        dict:    
    """
    #   interaction (train_random )
    unlearn_interactions = data_generator.train_random[['user', 'item', 'label']].values
    
    ranking_metrics = {
        'unlearn_positive_rank_drop_ratio': 0.0,
        'unlearn_positive_avg_rank_original': 0.0,
        'unlearn_positive_avg_rank_unlearned': 0.0,
        'unlearn_positive_rank_change': 0.0,
    }
    
    if len(unlearn_interactions) == 0:
        return ranking_metrics
    
    #      (  )
    user_interactions = {}
    for user_id, item_id in data_generator.train[['user', 'item']].values:
        if user_id not in user_interactions:
            user_interactions[user_id] = set()
        user_interactions[user_id].add(item_id)
    
    #  unlearn interactions 
    user_unlearn_items = {}
    for user_id, item_id, label in unlearn_interactions:
        if user_id not in user_interactions:
            continue
        if user_id not in user_unlearn_items:
            user_unlearn_items[user_id] = []
        user_unlearn_items[user_id].append(item_id)
    
    #      
    unlearn_positive_original_ranks = []
    unlearn_positive_unlearned_ranks = []
    unlearn_positive_rank_drops_count = 0
    
    #     
    for user_id, unlearn_items in user_unlearn_items.items():
        if not unlearn_items:
            continue
            
        user_items = list(user_interactions[user_id])
        
        try:
            #        (SISA )
            user_scores_original = original_model.batch_rating_agg([user_id], user_items).flatten()
            user_scores_unlearned = unlearned_model.batch_rating_agg([user_id], user_items).flatten()
        except Exception as e:
            print(f"Warning: Score calculation failed for user {user_id}: {e}")
            continue
        
        #   ( )
        original_ranks_user = np.argsort(np.argsort(-user_scores_original)) + 1
        unlearned_ranks_user = np.argsort(np.argsort(-user_scores_unlearned)) + 1
        
        #  ID   (  )
        item_to_idx = {item: idx for idx, item in enumerate(user_items)}
        
        #  unlearn    
        for item_id in unlearn_items:
            if item_id not in item_to_idx:
                continue
                
            item_idx = item_to_idx[item_id]
            original_rank = original_ranks_user[item_idx]
            unlearned_rank = unlearned_ranks_user[item_idx]
            
            unlearn_positive_original_ranks.append(original_rank)
            unlearn_positive_unlearned_ranks.append(unlearned_rank)
            
            #   
            if original_rank - unlearned_rank < 0:
                unlearn_positive_rank_drops_count += 1
    
    #  
    if unlearn_positive_original_ranks:
        ranking_metrics['unlearn_positive_rank_drop_ratio'] = unlearn_positive_rank_drops_count / len(unlearn_positive_original_ranks)
        ranking_metrics['unlearn_positive_avg_rank_original'] = np.mean(unlearn_positive_original_ranks)
        ranking_metrics['unlearn_positive_avg_rank_unlearned'] = np.mean(unlearn_positive_unlearned_ranks)
        ranking_metrics['unlearn_positive_rank_change'] = np.mean(unlearn_positive_original_ranks) - np.mean(unlearn_positive_unlearned_ranks)
    
    #   
    ranking_metrics['overall_effectiveness'] = ranking_metrics['unlearn_positive_rank_drop_ratio']
    
    return ranking_metrics


def compute_retraining_based_metrics_sisa(retraining_model, unlearning_model, data_generator, k_list=[20, 50, 100]):
    """
    SISA   Retraining  ground truth  Unlearning  Recall@K, NDCG@K  
    Retraining       top-k 
    
    Args:
        retraining_model: Retraining SISA  (ground truth)
        unlearning_model: Unlearning SISA  ( )
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
    
    #        
    user_train_items = {}
    for user_id, item_id in data_generator.train[['user', 'item']].values:
        if user_id not in user_train_items:
            user_train_items[user_id] = set()
        user_train_items[user_id].add(item_id)
    
    #    
    for user_id in range(data_generator.n_users):
        #     
        all_items = np.arange(data_generator.n_items)
        
        try:
            retraining_scores = retraining_model.batch_rating([user_id], all_items).flatten()
            unlearning_scores = unlearning_model.batch_rating_agg([user_id], all_items).flatten()
        except Exception as e:
            print(f"Warning: Score calculation failed for user {user_id}: {e}")
            continue
        
        #      (   )
        train_items = user_train_items.get(user_id, set())
        mask = np.ones(data_generator.n_items, dtype=bool)
        for item_id in train_items:
            mask[item_id] = False
        
        #  
        masked_retraining_scores = retraining_scores.copy()
        masked_unlearning_scores = unlearning_scores.copy()
        masked_retraining_scores[~mask] = -np.inf
        masked_unlearning_scores[~mask] = -np.inf
        
        # Top-max_k  (  )
        retraining_top_max_k_indices = np.argsort(-masked_retraining_scores)[:max_k]
        retraining_top_max_k_items = [all_items[i] for i in retraining_top_max_k_indices]
        
        unlearning_top_max_k_indices = np.argsort(-masked_unlearning_scores)[:max_k]
        unlearning_top_k_items = [all_items[i] for i in unlearning_top_max_k_indices]
        
        #  K  
        for k in k_list:
            # Top-k  
            retraining_top_k_items = retraining_top_max_k_items[:k]
            unlearning_top_k_items = unlearning_top_k_items[:k]
            
            # Recall@K 
            relevant_items = set(retraining_top_k_items)
            recommended_items = set(unlearning_top_k_items)
            recall = len(relevant_items.intersection(recommended_items)) / len(relevant_items) if len(relevant_items) > 0 else 0
            metrics[f'recall_at_{k}'].append(recall)
            
            # NDCG@K 
            dcg = 0
            idcg = 0
            
            # DCG 
            for i, item_id in enumerate(unlearning_top_k_items):
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
