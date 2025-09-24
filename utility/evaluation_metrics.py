import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def calculate_recall_at_k(predictions, ground_truth, k_list=[20, 50, 100]):
    """
    Calculate Recall@K for given k values
    
    Args:
        predictions: predicted scores for all items
        ground_truth: ground truth labels (1 for positive, 0 for negative)
        k_list: list of k values to calculate recall
    
    Returns:
        dict: recall values for each k
    """
    recalls = {}
    
    # Sort predictions in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    
    for k in k_list:
        # Get top-k items
        top_k_indices = sorted_indices[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = np.sum(ground_truth[top_k_indices])
        
        # Total relevant items
        total_relevant = np.sum(ground_truth)
        
        # Calculate recall
        if total_relevant > 0:
            recall = relevant_in_top_k / total_relevant
        else:
            recall = 0.0
            
        recalls[f'Recall@{k}'] = recall
    
    return recalls


def calculate_ndcg_at_k(predictions, ground_truth, k_list=[20, 50, 100]):
    """
    Calculate NDCG@K for given k values
    
    Args:
        predictions: predicted scores for all items
        ground_truth: ground truth labels (1 for positive, 0 for negative)
        k_list: list of k values to calculate NDCG
    
    Returns:
        dict: NDCG values for each k
    """
    ndcgs = {}
    
    # Sort predictions in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    
    for k in k_list:
        # Get top-k items
        top_k_indices = sorted_indices[:k]
        
        # Get relevance scores for top-k items
        relevance_scores = ground_truth[top_k_indices]
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / np.log2(i + 2)  # log2(i+2) because i starts from 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = np.sort(ground_truth)[::-1][:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        # Calculate NDCG
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
            
        ndcgs[f'NDCG@{k}'] = ndcg
    
    return ndcgs


def calculate_hit_ratio_at_k(predictions, ground_truth, k_list=[20, 50, 100]):
    """
    Calculate Hit Ratio@K for given k values
    
    Args:
        predictions: predicted scores for all items
        ground_truth: ground truth labels (1 for positive, 0 for negative)
        k_list: list of k values to calculate hit ratio
    
    Returns:
        dict: hit ratio values for each k
    """
    hit_ratios = {}
    
    # Sort predictions in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    
    for k in k_list:
        # Get top-k items
        top_k_indices = sorted_indices[:k]
        
        # Check if any relevant item is in top-k
        has_relevant = np.any(ground_truth[top_k_indices])
        
        hit_ratios[f'HitRatio@{k}'] = 1.0 if has_relevant else 0.0
    
    return hit_ratios


def calculate_all_metrics(predictions, ground_truth, k_list=[20, 50, 100]):
    """
    Calculate all evaluation metrics
    
    Args:
        predictions: predicted scores for all items
        ground_truth: ground truth labels (1 for positive, 0 for negative)
        k_list: list of k values to calculate metrics
    
    Returns:
        dict: all metric values
    """
    metrics = {}
    
    # Calculate AUC
    metrics['AUC'] = roc_auc_score(ground_truth, predictions)
    
    # Calculate Recall@K
    recall_metrics = calculate_recall_at_k(predictions, ground_truth, k_list)
    metrics.update(recall_metrics)
    
    # Calculate NDCG@K
    ndcg_metrics = calculate_ndcg_at_k(predictions, ground_truth, k_list)
    metrics.update(ndcg_metrics)
    
    # Calculate Hit Ratio@K
    hit_metrics = calculate_hit_ratio_at_k(predictions, ground_truth, k_list)
    metrics.update(hit_metrics)
    
    return metrics


def evaluate_model_per_user(model, data_generator, k_list=[20, 50, 100]):
    """
    Evaluate model performance per user
    
    Args:
        model: trained model
        data_generator: data generator object
        k_list: list of k values to calculate metrics
    
    Returns:
        dict: average metrics across all users
    """
    test_data = data_generator.test[['user', 'item', 'label']].values
    valid_data = data_generator.valid[['user', 'item', 'label']].values
    
    # Get unique users
    all_users = np.unique(np.concatenate([test_data[:, 0], valid_data[:, 0]]))
    
    all_metrics = {
        'AUC': [],
        'Recall@20': [], 'Recall@50': [], 'Recall@100': [],
        'NDCG@20': [], 'NDCG@50': [], 'NDCG@100': [],
        'HitRatio@20': [], 'HitRatio@50': [], 'HitRatio@100': []
    }
    
    for user in all_users:
        # Get all items for this user
        all_items = np.arange(data_generator.n_items)
        
        # Get predictions for this user
        user_predictions = model.predict(np.full_like(all_items, user), all_items)
        
        # Get ground truth for this user
        user_ground_truth = np.zeros(data_generator.n_items)
        
        # Set positive items to 1
        user_test_items = test_data[test_data[:, 0] == user, 1]
        user_valid_items = valid_data[valid_data[:, 0] == user, 1]
        user_positive_items = np.concatenate([user_test_items, user_valid_items])
        user_ground_truth[user_positive_items] = 1
        
        # Calculate metrics for this user
        user_metrics = calculate_all_metrics(user_predictions, user_ground_truth, k_list)
        
        # Append to all metrics
        for metric_name, value in user_metrics.items():
            all_metrics[metric_name].append(value)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name, values in all_metrics.items():
        avg_metrics[metric_name] = np.mean(values)
    
    return avg_metrics


def print_evaluation_results(metrics, prefix=""):
    """
    Print evaluation results in a formatted way
    
    Args:
        metrics: dictionary of metrics
        prefix: prefix for printing
    """
    print(f"{prefix}Evaluation Results:")
    print(f"{prefix}AUC: {metrics['AUC']:.4f}")
    
    for k in [20, 50, 100]:
        print(f"{prefix}Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
        print(f"{prefix}NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
        print(f"{prefix}HitRatio@{k}: {metrics[f'HitRatio@{k}']:.4f}")
    print() 