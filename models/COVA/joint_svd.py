import numpy as np
import torch
import math

def joint_svd_unlearning(original_interaction_matrix,
                         ideal_interaction_matrix, 
                         predict_prob, alpha, 
                         beta, num_dim, num_iter,
                         stacking_method='vertical'):
    device = original_interaction_matrix.device

    R_original = original_interaction_matrix
    R_ideal = ideal_interaction_matrix
    R_pred = predict_prob
    n_users, n_items = R_original.shape
    
    R_combined = torch.cat([R_original, R_ideal, R_pred], dim=0)
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_combined, q=num_dim, niter=num_iter)
    
    U1 = U_joint[:n_users]
    U2 = U_joint[n_users:2*n_users]
    U3 = U_joint[2*n_users:3*n_users]

    S_shared = S_joint
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
        
    sqrt_S = torch.sqrt(S_shared)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)          
    
    return user_embed, item_embed

def joint_svd_unlearning_v2(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48, alpha=0.5):
    R_weighted = torch.cat([
        torch.sqrt(torch.tensor(w1)) * R_original,
        torch.sqrt(torch.tensor(w2)) * R_ideal,
        torch.sqrt(torch.tensor(w3)) * R_pred
    ], dim=0)
    
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    n = R_original.shape[0]
    U1 = U_joint[:n] / torch.sqrt(torch.tensor(w1))
    U2 = U_joint[n:2*n] / torch.sqrt(torch.tensor(w2))
    U3 = U_joint[2*n:3*n] / torch.sqrt(torch.tensor(w3))

    # Orthogonality-aware correction
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U

    # Apply orthogonality constraint
    # U3_corrected = U3_corrected - lambda_orth * U1 @ (U1.T @ U3_corrected)
    
    # Normalize for stability
    # U3_corrected = torch.nn.functional.normalize(U3_corrected, dim=1)
    
    # Compute final embeddings
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    return user_embed, item_embed


def joint_svd_unlearning_v3(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48):
    
    ## Original
    positive_mask = (R_original == 1)
    negative_mask = (R_original == -1)
    unobserved_mask = (R_original == 0)
    
    pos_counts = positive_mask.sum(dim=1, keepdim=True)
    neg_counts = negative_mask.sum(dim=1, keepdim=True)
    unobs_counts = unobserved_mask.sum(dim=1, keepdim=True)

    pos = R_pred.max(dim=1, keepdim=True)[0]
    neg = R_pred.min(dim=1, keepdim=True)[0]
    unobs = (pos + neg) / 2
    R_original_adjusted = torch.where(positive_mask, pos, torch.where(negative_mask, neg, unobs)) * w1
    
    ## Ideal
    positive_mask = (R_ideal == 1)
    negative_mask = (R_ideal == -1)
    unobserved_mask = (R_ideal == 0)
    
    R_ideal_adjusted = torch.where(positive_mask, pos, torch.where(negative_mask, neg, unobs)) * w1
    
    del positive_mask, negative_mask, unobserved_mask, pos_counts, neg_counts, unobs_counts
    torch.cuda.empty_cache()
    
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0)
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    del R_weighted, U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


def joint_svd_unlearning_v3_mean(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48, alpha=0.5):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    
    # Positive interaction   ( )
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))
    
    # Unobserved interaction   ( )
    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))
    
    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1
    
    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    
    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w1
    
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


def joint_svd_unlearning_v3_mean2(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48, alpha=0.5):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    
    # Positive interaction   ( )
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))
    
    # Unobserved interaction   ( )
    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))
    
    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1
    
    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    
    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w2
    
    R_pred = R_pred * w1
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


def joint_svd_unlearning_v3_mean3(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48, alpha=0.5):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0
    
    # Positive interaction   ( )
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))
    
    # Unobserved interaction   ( )
    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))
    
    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1
    
    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    # R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w2
    
    unlearn_weight = torch.min(R_pred, dim=1, keepdim=True)[0]
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean)) * w2
    
    R_pred = R_pred * w1

    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


def joint_svd_unlearning_v3_mean4(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48, alpha=0.5):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0
    
    # Positive interaction   ( )
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))
    
    # Unobserved interaction   ( )
    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))
    
    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1

    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    unobs_var = (((R_pred - unobs_mean) ** 2) * unobserved_mask).sum(dim=1, keepdim=True) / unobs_count
    eps = 1e-8
    unobs_std = torch.sqrt(unobs_var + eps)
    unlearn_weight = unobs_mean - unobs_std
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean)) * w2
    
    R_pred = R_pred * w1
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed

def joint_svd_unlearning_v3_minmax(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48):
    
    positive_mask = (R_original == 1)
    
    pos = torch.max(R_pred, dim=1, keepdim=True)[0]
    unobs = torch.min(R_pred, dim=1, keepdim=True)[0]

    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos, unobs) * w1

    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    
    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos, unobs) * w1
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


def joint_svd_unlearning_v3_gaussian(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    
    #  /: 0       
    R_pred = R_pred.float()
    pos_mask = positive_mask.float()
    unobs_mask = unobserved_mask.float()

    pos_count = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    unobs_count = unobs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    pos_sum = (R_pred * pos_mask).sum(dim=1, keepdim=True)
    unobs_sum = (R_pred * unobs_mask).sum(dim=1, keepdim=True)

    pos_mean = pos_sum / pos_count
    unobs_mean = unobs_sum / unobs_count

    pos_var = (((R_pred - pos_mean) ** 2) * pos_mask).sum(dim=1, keepdim=True) / pos_count
    unobs_var = (((R_pred - unobs_mean) ** 2) * unobs_mask).sum(dim=1, keepdim=True) / unobs_count

    eps = 1e-8
    pos_std = torch.sqrt(pos_var + eps)
    unobs_std = torch.sqrt(unobs_var + eps)
    
    pos_adjusted = pos_mean + pos_std * torch.randn_like(pos_mean)
    unobs_adjusted = unobs_mean + unobs_std * torch.randn_like(unobs_mean)

    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_adjusted, unobs_adjusted) * w1

    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)

    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_adjusted, unobs_adjusted) * w1
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


######################### User weight Version #########################

def joint_svd_unlearning_v3_mean_user_weight(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    
    # Positive interaction   ( )
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))
    
    # Unobserved interaction   ( )
    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))
    
    # weight
    original_counts = R_original.sum(dim=1, keepdim=True).float()
    unlearned_counts = (R_original - R_ideal).sum(dim=1, keepdim=True).float()
    unlearned_ratios = unlearned_counts / original_counts * w2
    w1_for_pred = (1+unlearned_ratios)
    w1 = (1 + unlearned_ratios) * w1
    
    
    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1
    
    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    
    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w1
    
    R_pred = R_pred * w1_for_pred
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed

def joint_svd_unlearning_v3_minmax_user_weight(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48):
    
    positive_mask = (R_original == 1)
    
    pos = torch.max(R_pred, dim=1, keepdim=True)[0]
    unobs = torch.min(R_pred, dim=1, keepdim=True)[0]

    # weight
    original_counts = R_original.sum(dim=1, keepdim=True).float()
    unlearned_counts = (R_original - R_ideal).sum(dim=1, keepdim=True).float()
    unlearned_ratios = unlearned_counts / original_counts * w2
    w1_for_pred = (1+unlearned_ratios)
    w1 = (1 + unlearned_ratios) * w1

    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos, unobs) * w1
    
    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)
    
    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos, unobs) * w1
    
    R_pred = R_pred * w1_for_pred
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed


def joint_svd_unlearning_v3_gaussian_user_weight(R_original, R_ideal, R_pred, 
                                  w1=1.0, w2=1.0, w3=5.0, 
                                  beta=5, num_iter=20, 
                                  num_dim=48):
    
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    
    pos_mean = torch.mean(R_pred * positive_mask.float(), dim=1, keepdim=True)
    unobs_mean = torch.mean(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    pos_std = torch.std(R_pred * positive_mask.float(), dim=1, keepdim=True)
    unobs_std = torch.std(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    
    pos_adjusted = pos_mean + pos_std * torch.randn_like(pos_mean)
    unobs_adjusted = unobs_mean + unobs_std * torch.randn_like(unobs_mean)

    # weight
    original_counts = R_original.sum(dim=1, keepdim=True).float()
    unlearned_counts = (R_original - R_ideal).sum(dim=1, keepdim=True).float()
    unlearned_ratios = unlearned_counts / original_counts * w2
    w1_for_pred = (1+unlearned_ratios)
    w1 = (1 + unlearned_ratios) * w1

    #  original matrix  (in-place )
    R_original_adjusted = torch.where(positive_mask, pos_adjusted, unobs_adjusted) * w1

    ## Ideal matrix adjustment (  )
    positive_mask_ideal = (R_ideal == 1)

    # Ideal matrix   
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_adjusted, unobs_adjusted) * w1
    
    R_pred = R_pred * w1_for_pred
    
    # Joint SVD 
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    #  
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n] 
    U2 = U_joint[n:2*n] 
    U3 = U_joint[2*n:3*n]

    # Correction  (in-place )
    Delta_U = U2 - U1
    U3_corrected = U3 + beta * Delta_U
    
    #    ( )
    sqrt_S = torch.sqrt(S_joint)
    user_embed = U3_corrected @ torch.diag(sqrt_S)
    item_embed = V_joint @ torch.diag(sqrt_S)
    
    #  
    del U_joint, S_joint, V_joint, U1, U2, U3, Delta_U, U3_corrected, sqrt_S
    torch.cuda.empty_cache()
    
    return user_embed, item_embed

def joint_svd_unlearning_v4(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 5.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    """
    Joint SVD  Original-   .
    - X-(X = U @ diag(sqrt(S)))   ,  (Original)   
      (Ideal)  .

    (   ):
      X_new = X3 + alpha * (Delta1_perp) - beta * (Delta2_perp)
      where
        Xk = Uk @ diag(sqrt(S)),
        Delta1 = X2 - X1, Delta2 = X3 - X1,
        Delta*_perp = Delta* - proj_{X1}(Delta*), () 

    :
      user_embed (X_new), item_embed (V @ diag(sqrt(S)))
    """
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1

    positive_mask_ideal = (R_ideal == 1)
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w2

    R_pred = R_pred * w1

    # Joint SVD
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    # Cleanup big tensor
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    X1 = U1
    X2 = U2
    X3 = U3
    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)
    x1_norm_sq = (X1 * X1).sum(dim=1, keepdim=True) + eps

    Delta1 = X2 - X1
    Delta2 = X3 - X1

    coeff1 = (Delta1 * X1).sum(dim=1, keepdim=True) / x1_norm_sq
    coeff2 = (Delta2 * X1).sum(dim=1, keepdim=True) / x1_norm_sq

    Delta1_perp = Delta1 - coeff1 * X1
    Delta2_perp = Delta2 - coeff2 * X1

    # Orthogonal residual correction (anchor at X3)
    X_new = X3 + beta * Delta1_perp - alpha * Delta2_perp

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, X1, X2, X3, Delta1, Delta2, Delta1_perp, Delta2_perp, coeff1, coeff2, x1_norm_sq
    torch.cuda.empty_cache()

    user_embed = X_new @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed

def joint_svd_unlearning_v4_wonorm(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 5.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    """
    Joint SVD  Original-   .
    - X-(X = U @ diag(sqrt(S)))   ,  (Original)   
      (Ideal)  .

    (   ):
      X_new = X3 + alpha * (Delta1_perp) - beta * (Delta2_perp)
      where
        Xk = Uk @ diag(sqrt(S)),
        Delta1 = X2 - X1, Delta2 = X3 - X1,
        Delta*_perp = Delta* - proj_{X1}(Delta*), () 

    :
      user_embed (X_new), item_embed (V @ diag(sqrt(S)))
    """
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1

    positive_mask_ideal = (R_ideal == 1)
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w2

    R_pred = R_pred * w1

    # Joint SVD
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    # Cleanup big tensor
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    X1 = U1 
    X2 = U2
    X3 = U3
    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)

    Delta1 = X2 - X1
    Delta2 = X3 - X1

    # Orthogonal residual correction (anchor at X3)
    X_new = X3 + beta * Delta1 - alpha * Delta2

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, X1, X2, X3, Delta1, Delta2
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed


def joint_svd_unlearning_v4_min(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 5.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    """
    Joint SVD  Original-   .
    - X-(X = U @ diag(sqrt(S)))   ,  (Original)   
      (Ideal)  .

    (   ):
      X_new = X3 + alpha * (Delta1_perp) - beta * (Delta2_perp)
      where
        Xk = Uk @ diag(sqrt(S)),
        Delta1 = X2 - X1, Delta2 = X3 - X1,
        Delta*_perp = Delta* - proj_{X1}(Delta*), () 

    :
      user_embed (X_new), item_embed (V @ diag(sqrt(S)))
    """
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1

    positive_mask_ideal = (R_ideal == 1)
    # R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, unobs_mean) * w2
    
    unlearn_weight = torch.min(R_pred, dim=1, keepdim=True)[0]
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean)) * w2
    

    R_pred = R_pred * w1

    # Joint SVD
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    # Cleanup big tensor
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    X1 = U1
    X2 = U2
    X3 = U3
    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)
    x1_norm_sq = (X1 * X1).sum(dim=1, keepdim=True) + eps

    Delta1 = X2 - X1
    Delta2 = X3 - X1

    coeff1 = (Delta1 * X1).sum(dim=1, keepdim=True) / x1_norm_sq
    coeff2 = (Delta2 * X1).sum(dim=1, keepdim=True) / x1_norm_sq

    Delta1_perp = Delta1 - coeff1 * X1
    Delta2_perp = Delta2 - coeff2 * X1

    # Orthogonal residual correction (anchor at X3)
    X_new = X3 + beta * Delta1_perp - alpha * Delta2_perp

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, X1, X2, X3, Delta1, Delta2, Delta1_perp, Delta2_perp, coeff1, coeff2, x1_norm_sq
    torch.cuda.empty_cache()

    user_embed = X_new @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed

def joint_svd_unlearning_v4_min_wonorm(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 5.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean) * w1

    positive_mask_ideal = (R_ideal == 1)
    unlearn_weight = torch.min(R_pred, dim=1, keepdim=True)[0]
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean)) * w2

    R_pred = R_pred * w1

    # Joint SVD
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    # Cleanup big tensor
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    X1 = U1 
    X2 = U2
    X3 = U3
    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)

    Delta1 = X2 - X1
    Delta2 = X3 - X1

    # Orthogonal residual correction (anchor at X3)
    X_new = X3 + beta * Delta1 - alpha * Delta2

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, X1, X2, X3, Delta1, Delta2
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed


def joint_svd_unlearning_v5(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean)

    positive_mask_ideal = (R_ideal == 1)

    ## User-unlearned weight
    unobs_squared_sum = torch.sum((R_pred * unobserved_mask.float())**2, dim=1, keepdim=True)
    unobs_variance = torch.where(unobs_count > 1, 
                                (unobs_squared_sum - unobs_sum**2 / unobs_count) / (unobs_count - 1), 
                                torch.zeros_like(unobs_sum))
    unobs_std = torch.sqrt(torch.clamp(unobs_variance, min=1e-8)) 
    
    unlearn_count = unlearn_mask.sum(dim=1, keepdim=True).float()
    user_weight = (1 + (unlearn_count/pos_count))**w1
    unlearn_weight = unobs_mean - unobs_std * user_weight * w2

    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean))
    ideal_weight = math.sqrt(w3)

    # Joint SVD
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted * ideal_weight, R_pred], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    # Cleanup big tensor
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)

    Delta1 = U2 - U1 # Unlearning Impact
    Delta2 = U3 - U1 # Collaborative Impact of Original

    # Orthogonal residual correction (anchor at X3)
    X_new = U3 + beta * Delta1 - alpha * Delta2

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed
  
  
  
def joint_svd_unlearning_v6(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean)

    positive_mask_ideal = (R_ideal == 1)

    ## User-unlearned weight
    unobs_squared_sum = torch.sum((R_pred * unobserved_mask.float())**2, dim=1, keepdim=True)
    unobs_variance = torch.where(unobs_count > 1, 
                                (unobs_squared_sum - unobs_sum**2 / unobs_count) / (unobs_count - 1), 
                                torch.zeros_like(unobs_sum))
    unobs_std = torch.sqrt(torch.clamp(unobs_variance, min=1e-8)) 

    unlearn_count = unlearn_mask.sum(dim=1, keepdim=True).float()
    user_weight = (1 + (unlearn_count/pos_count))**w1
    unlearn_weight = unobs_mean - unobs_std * user_weight * w2

    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean))
    ideal_weight = math.sqrt(w3)

    # Joint SVD
    R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred * ideal_weight], dim=0).cuda()
    U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)

    # Cleanup big tensor
    del R_weighted
    torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)
    Delta1 = U2 - U1 # Unlearning Impact

    # Orthogonal residual correction (anchor at X3)
    X_new = U3 + beta * Delta1

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed



def joint_svd_unlearning_v7(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean)
    positive_mask_ideal = (R_ideal == 1)

    unlearn_weight = torch.min(R_pred, dim=1, keepdim=True)[0]

    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean))
    
    # Try original path first; on OOM, fallback to randomized low-rank without building [A;B;C]
    try:
        torch.cuda.empty_cache()
        R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).to(device='cuda', dtype=torch.float32)
        U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)
        del R_weighted
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A1 = R_original_adjusted
        A2 = R_ideal_adjusted
        A3 = R_pred

        ncols = A1.shape[1] 
        # Omega on GPU (fp32 for QR stability)
        Omega = torch.randn(ncols, num_dim, device=device, dtype=torch.float32)

        # Y = [A1@Omega; A2@Omega; A3@Omega]
        Y1 = A1.to(device=device, dtype=torch.float32) @ Omega
        Y2 = A2.to(device=device, dtype=torch.float32) @ Omega
        Y3 = A3.to(device=device, dtype=torch.float32) @ Omega
        X = torch.cat([Y1, Y2, Y3], dim=0)
        del Y1, Y2, Y3
        torch.cuda.empty_cache()

        # Orthonormal basis Q of Y
        Q = torch.linalg.qr(X).Q # (m, q)

        # Power iterations (Halko 4.4)
        for _ in range(num_iter):
            n_users = A1.shape[0]
            Q1 = Q[:n_users]
            Q2 = Q[n_users:2*n_users]
            Q3 = Q[2*n_users:3*n_users]

            # X = A^T @ Q
            X = A1.to(device=device, dtype=torch.float32).T @ Q1
            X += A2.to(device=device, dtype=torch.float32).T @ Q2
            X += A3.to(device=device, dtype=torch.float32).T @ Q3
            del Q1, Q2, Q3
            torch.cuda.empty_cache()
            Q = torch.linalg.qr(X).Q
            
            del X
            torch.cuda.empty_cache()
            
            # X = A @ Q
            X1 = A1.to(device=device, dtype=torch.float32) @ Q
            X2 = A2.to(device=device, dtype=torch.float32) @ Q
            X3 = A3.to(device=device, dtype=torch.float32) @ Q
            X = torch.cat([X1, X2, X3], dim=0)
            del X1, X2, X3, Q
            torch.cuda.empty_cache()

            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

        # B = Q^T @ A = sum_i Q_i^T @ A_i
        n = A1.shape[0]
        Q1 = Q[:n]
        Q2 = Q[n:2*n]
        Q3 = Q[2*n:3*n]

        B = Q1.T @ A1.to(device=device, dtype=torch.float32)
        B += Q2.T @ A2.to(device=device, dtype=torch.float32)
        B += Q3.T @ A3.to(device=device, dtype=torch.float32)

        # SVD of small B (l x m)
        Ub, S_joint, Vh = torch.linalg.svd(B, full_matrices=False)
        V_joint = Vh.mH
        del B, Vh, Q1, Q2, Q3
        torch.cuda.empty_cache()

        U_joint = Q.matmul(Ub)
        del Q, Ub
        torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)
    Delta1 = U2 - U1 # Unlearning Impact
    Delta2 = U3 - U1 # Collaborative Impact of Original

    # Orthogonal residual correction (anchor at X3)
    X_new = U3 + beta * Delta1 - alpha * Delta2

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed


def joint_svd_unlearning_v8(R_original, R_ideal, R_pred, num_chunks=3,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 10, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean-based adjustment (same as v3_mean2)
    pos_sum = torch.sum(R_pred * positive_mask.float(), dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1, keepdim=True).float()
    pos_mean = torch.where(pos_count > 0, pos_sum / pos_count, torch.zeros_like(pos_sum))

    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.sum(dim=1, keepdim=True).float()
    unobs_mean = torch.where(unobs_count > 0, unobs_sum / unobs_count, torch.zeros_like(unobs_sum))

    R_original_adjusted = torch.where(positive_mask, pos_mean, unobs_mean)
    positive_mask_ideal = (R_ideal == 1)

    unlearn_weight = torch.min(R_pred, dim=1, keepdim=True)[0]

    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_mean, torch.where(unlearn_mask, unlearn_weight, unobs_mean))
    
    # Try original path first; on OOM, fallback to randomized low-rank without building [A;B;C]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------
    # Chunked helpers (row-wise)
    # ------------------------------
    def _rows_chunk_size(n_rows: int, chunks: int) -> int:
        return (n_rows + chunks - 1) // chunks if chunks > 0 else n_rows

    def matmul_rows_in_chunks(A_cpu: torch.Tensor, B_dev: torch.Tensor, chunks: int) -> torch.Tensor:
        """
        Compute A @ B by streaming A in row-chunks to device.
        - A_cpu: (n, m) on CPU
        - B_dev: (m, k) on device
        Returns: (n, k) on device
        """
        n_rows = A_cpu.shape[0]
        out = torch.empty(n_rows, B_dev.shape[1], device=B_dev.device, dtype=torch.float32)
        step = _rows_chunk_size(n_rows, chunks)
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            a_chunk = A_cpu[start:end].to(device=B_dev.device, dtype=torch.float32)
            out[start:end] = a_chunk @ B_dev
            del a_chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
        return out

    def matmul_AT_Q_in_chunks(A_cpu: torch.Tensor, Q_dev: torch.Tensor, chunks: int) -> torch.Tensor:
        """
        Compute A^T @ Q by streaming rows of A (and corresponding rows of Q) in chunks.
        - A_cpu: (n, m) on CPU
        - Q_dev: (n, q) on device
        Returns: (m, q) on device
        """
        n_rows, n_cols = A_cpu.shape
        q_dim = Q_dev.shape[1]
        out = torch.zeros(n_cols, q_dim, device=Q_dev.device, dtype=torch.float32)
        step = _rows_chunk_size(n_rows, chunks)
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            a_chunk = A_cpu[start:end].to(device=Q_dev.device, dtype=torch.float32)  # (cs, m)
            q_chunk = Q_dev[start:end]  # (cs, q)
            out.add_(a_chunk.transpose(0, 1) @ q_chunk)  # (m, q)
            del a_chunk, q_chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
        return out

    def matmul_QT_A_in_chunks(QT_dev: torch.Tensor, A_cpu: torch.Tensor, chunks: int) -> torch.Tensor:
        """
        Compute Q^T @ A by streaming rows of A (and corresponding columns of Q^T) in chunks.
        - QT_dev: (q, n) on device
        - A_cpu: (n, m) on CPU
        Returns: (q, m) on device
        """
        n_rows, n_cols = A_cpu.shape
        q_dim = QT_dev.shape[0]
        out = torch.zeros(q_dim, n_cols, device=QT_dev.device, dtype=torch.float32)
        step = _rows_chunk_size(n_rows, chunks)
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            qt_chunk = QT_dev[:, start:end]  # (q, cs)
            a_chunk = A_cpu[start:end].to(device=QT_dev.device, dtype=torch.float32)  # (cs, m)
            out.add_(qt_chunk @ a_chunk)  # (q, m)
            del qt_chunk, a_chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
        return out

    A1 = R_original_adjusted
    A2 = R_ideal_adjusted
    A3 = R_pred

    ncols = A1.shape[1] 
    # Omega on GPU (fp32 for QR stability)
    Omega = torch.randn(ncols, num_dim, device=device, dtype=torch.float32)

    # Y = [A1@Omega; A2@Omega; A3@Omega]  (streamed row-chunks)
    Y1 = matmul_rows_in_chunks(A1, Omega, num_chunks)
    Y2 = matmul_rows_in_chunks(A2, Omega, num_chunks)
    Y3 = matmul_rows_in_chunks(A3, Omega, num_chunks)
    X = torch.cat([Y1, Y2, Y3], dim=0)
    del Y1, Y2, Y3
    torch.cuda.empty_cache()

    # Orthonormal basis Q of Y
    Q = torch.linalg.qr(X).Q # (m, q)

    # Power iterations (Halko 4.4)
    for _ in range(num_iter):
        n_users = A1.shape[0]
        Q1 = Q[:n_users]
        Q2 = Q[n_users:2*n_users]
        Q3 = Q[2*n_users:3*n_users]

        # X = A^T @ Q   (streamed row-chunks)
        X = matmul_AT_Q_in_chunks(A1, Q1, num_chunks)
        X += matmul_AT_Q_in_chunks(A2, Q2, num_chunks)
        X += matmul_AT_Q_in_chunks(A3, Q3, num_chunks)
        del Q1, Q2, Q3
        torch.cuda.empty_cache()
        Q = torch.linalg.qr(X).Q
        
        del X
        torch.cuda.empty_cache()
        
        # X = A @ Q  (streamed row-chunks)
        X1 = matmul_rows_in_chunks(A1, Q, num_chunks)
        X2 = matmul_rows_in_chunks(A2, Q, num_chunks)
        X3 = matmul_rows_in_chunks(A3, Q, num_chunks)
        X = torch.cat([X1, X2, X3], dim=0)
        del X1, X2, X3
        torch.cuda.empty_cache()

        Q = torch.linalg.qr(X).Q
        del X
        torch.cuda.empty_cache()

        # B = Q^T @ A = sum_i Q_i^T @ A_i  (streamed row-chunks)
        n = A1.shape[0]
        Q1 = Q[:n]
        Q2 = Q[n:2*n]
        Q3 = Q[2*n:3*n]

        B = matmul_QT_A_in_chunks(Q1.T, A1, num_chunks)
        B += matmul_QT_A_in_chunks(Q2.T, A2, num_chunks)
        B += matmul_QT_A_in_chunks(Q3.T, A3, num_chunks)

        # SVD of small B (l x m)
        Ub, S_joint, Vh = torch.linalg.svd(B, full_matrices=False)
        V_joint = Vh.mH
        del B, Vh, Q1, Q2, Q3
        torch.cuda.empty_cache()

        U_joint = Q.matmul(Ub)
        del Ub
        torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    # X-space embeddings
    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)

    Y = V_joint @ S_sqrt_mat

    # Original-orthogonal residuals (row-wise)
    Delta1 = U2 - U1 # Unlearning Impact
    Delta2 = U3 - U1 # Collaborative Impact of Original

    # Orthogonal residual correction (anchor at X3)
    X_new = U3 + beta * Delta1 - alpha * Delta2

    # Optional: memory cleanup of intermediates
    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed




def joint_svd_unlearning_v7_mean_std(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Mean and std (masked)
    R_pred = R_pred.float()
    pos_mask = positive_mask.float()
    unobs_mask = unobserved_mask.float()

    pos_count = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    unobs_count = unobs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    pos_sum = (R_pred * pos_mask).sum(dim=1, keepdim=True)
    unobs_sum = (R_pred * unobs_mask).sum(dim=1, keepdim=True)

    pos_mean = pos_sum / pos_count
    unobs_mean = unobs_sum / unobs_count

    pos_var = (((R_pred - pos_mean) ** 2) * pos_mask).sum(dim=1, keepdim=True) / pos_count
    unobs_var = (((R_pred - unobs_mean) ** 2) * unobs_mask).sum(dim=1, keepdim=True) / unobs_count

    pos_std = torch.sqrt(pos_var + eps)
    unobs_std = torch.sqrt(unobs_var + eps)

    # Adjusted matrices
    pos_adjusted = pos_mean + pos_std
    unobs_adjusted = unobs_mean
    unlearn_weight = unobs_mean - unobs_std

    R_original_adjusted = torch.where(positive_mask, pos_adjusted, unobs_adjusted)

    positive_mask_ideal = (R_ideal == 1)
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos_adjusted, torch.where(unlearn_mask, unlearn_weight, unobs_adjusted))

    # Try original path first; on OOM, fallback to randomized low-rank without building [A;B;C]
    try:
        torch.cuda.empty_cache()
        R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).to(device=('cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.float32)
        U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)
        del R_weighted
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A1 = R_original_adjusted
        A2 = R_ideal_adjusted
        A3 = R_pred

        ncols = A1.shape[1]
        Omega = torch.randn(ncols, num_dim, device=device, dtype=torch.float32)

        Y1 = A1.to(device=device, dtype=torch.float32) @ Omega
        Y2 = A2.to(device=device, dtype=torch.float32) @ Omega
        Y3 = A3.to(device=device, dtype=torch.float32) @ Omega
        X = torch.cat([Y1, Y2, Y3], dim=0)
        del Y1, Y2, Y3
        torch.cuda.empty_cache()

        Q = torch.linalg.qr(X).Q

        for _ in range(num_iter):
            n_users = A1.shape[0]
            Q1 = Q[:n_users]
            Q2 = Q[n_users:2*n_users]
            Q3 = Q[2*n_users:3*n_users]

            X = A1.to(device=device, dtype=torch.float32).T @ Q1
            X += A2.to(device=device, dtype=torch.float32).T @ Q2
            X += A3.to(device=device, dtype=torch.float32).T @ Q3
            del Q1, Q2, Q3
            torch.cuda.empty_cache()
            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

            X1 = A1.to(device=device, dtype=torch.float32) @ Q
            X2 = A2.to(device=device, dtype=torch.float32) @ Q
            X3 = A3.to(device=device, dtype=torch.float32) @ Q
            X = torch.cat([X1, X2, X3], dim=0)
            del X1, X2, X3, Q
            torch.cuda.empty_cache()

            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

        n = A1.shape[0]
        Q1 = Q[:n]
        Q2 = Q[n:2*n]
        Q3 = Q[2*n:3*n]

        B = Q1.T @ A1.to(device=device, dtype=torch.float32)
        B += Q2.T @ A2.to(device=device, dtype=torch.float32)
        B += Q3.T @ A3.to(device=device, dtype=torch.float32)

        Ub, S_joint, Vh = torch.linalg.svd(B, full_matrices=False)
        V_joint = Vh.mH
        del B, Vh, Q1, Q2, Q3
        torch.cuda.empty_cache()

        U_joint = Q.matmul(Ub)
        del Q, Ub
        torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)
    Y = V_joint @ S_sqrt_mat

    Delta1 = U2 - U1
    Delta2 = U3 - U1
    X_new = U3 + beta * Delta1 - alpha * Delta2

    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed


def joint_svd_unlearning_v7_max_min(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    # Stats
    pos = torch.max(R_pred, dim=1, keepdim=True)[0]
    unobs_sum = torch.sum(R_pred * unobserved_mask.float(), dim=1, keepdim=True)
    unobs_count = unobserved_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    unobs_mean = unobs_sum / unobs_count
    unlearn_weight = torch.min(R_pred, dim=1, keepdim=True)[0]

    R_original_adjusted = torch.where(positive_mask, pos, unobs_mean)

    positive_mask_ideal = (R_ideal == 1)
    R_ideal_adjusted = torch.where(positive_mask_ideal, pos, torch.where(unlearn_mask, unlearn_weight, unobs_mean))

    try:
        torch.cuda.empty_cache()
        R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).to(device='cuda', dtype=torch.float32)
        U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)
        del R_weighted
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A1 = R_original_adjusted
        A2 = R_ideal_adjusted
        A3 = R_pred

        ncols = A1.shape[1]
        Omega = torch.randn(ncols, num_dim, device=device, dtype=torch.float32)

        Y1 = A1.to(device=device, dtype=torch.float32) @ Omega
        Y2 = A2.to(device=device, dtype=torch.float32) @ Omega
        Y3 = A3.to(device=device, dtype=torch.float32) @ Omega
        X = torch.cat([Y1, Y2, Y3], dim=0)
        del Y1, Y2, Y3
        torch.cuda.empty_cache()

        Q = torch.linalg.qr(X).Q

        for _ in range(num_iter):
            n_users = A1.shape[0]
            Q1 = Q[:n_users]
            Q2 = Q[n_users:2*n_users]
            Q3 = Q[2*n_users:3*n_users]

            X = A1.to(device=device, dtype=torch.float32).T @ Q1
            X += A2.to(device=device, dtype=torch.float32).T @ Q2
            X += A3.to(device=device, dtype=torch.float32).T @ Q3
            del Q1, Q2, Q3
            torch.cuda.empty_cache()
            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

            X1 = A1.to(device=device, dtype=torch.float32) @ Q
            X2 = A2.to(device=device, dtype=torch.float32) @ Q
            X3 = A3.to(device=device, dtype=torch.float32) @ Q
            X = torch.cat([X1, X2, X3], dim=0)
            del X1, X2, X3, Q
            torch.cuda.empty_cache()

            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

        n = A1.shape[0]
        Q1 = Q[:n]
        Q2 = Q[n:2*n]
        Q3 = Q[2*n:3*n]

        B = Q1.T @ A1.to(device=device, dtype=torch.float32)
        B += Q2.T @ A2.to(device=device, dtype=torch.float32)
        B += Q3.T @ A3.to(device=device, dtype=torch.float32)

        Ub, S_joint, Vh = torch.linalg.svd(B, full_matrices=False)
        V_joint = Vh.mH
        del B, Vh, Q1, Q2, Q3
        torch.cuda.empty_cache()

        U_joint = Q.matmul(Ub)
        del Q, Ub
        torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)
    Y = V_joint @ S_sqrt_mat

    Delta1 = U2 - U1
    Delta2 = U3 - U1
    X_new = U3 + beta * Delta1 - alpha * Delta2

    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed


def joint_svd_unlearning_v7_binary(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    n = R_original.shape[0]
    ones = torch.ones((n, 1), device=R_pred.device, dtype=R_pred.dtype)
    zeros = torch.zeros((n, 1), device=R_pred.device, dtype=R_pred.dtype)
    neg_ones = -torch.ones((n, 1), device=R_pred.device, dtype=R_pred.dtype)

    R_original_adjusted = torch.where(positive_mask, ones, zeros)

    positive_mask_ideal = (R_ideal == 1)
    R_ideal_adjusted = torch.where(positive_mask_ideal, ones, torch.where(unlearn_mask, neg_ones, zeros))

    try:
        torch.cuda.empty_cache()
        R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).to(device='cuda', dtype=torch.float32)
        U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)
        del R_weighted
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A1 = R_original_adjusted
        A2 = R_ideal_adjusted
        A3 = R_pred

        ncols = A1.shape[1]
        Omega = torch.randn(ncols, num_dim, device=device, dtype=torch.float32)

        Y1 = A1.to(device=device, dtype=torch.float32) @ Omega
        Y2 = A2.to(device=device, dtype=torch.float32) @ Omega
        Y3 = A3.to(device=device, dtype=torch.float32) @ Omega
        X = torch.cat([Y1, Y2, Y3], dim=0)
        del Y1, Y2, Y3
        torch.cuda.empty_cache()

        Q = torch.linalg.qr(X).Q

        for _ in range(num_iter):
            n_users = A1.shape[0]
            Q1 = Q[:n_users]
            Q2 = Q[n_users:2*n_users]
            Q3 = Q[2*n_users:3*n_users]
            
            X = A1.to(device=device, dtype=torch.float32).T @ Q1
            X += A2.to(device=device, dtype=torch.float32).T @ Q2
            X += A3.to(device=device, dtype=torch.float32).T @ Q3
            del Q1, Q2, Q3
            torch.cuda.empty_cache()
            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

            X1 = A1.to(device=device, dtype=torch.float32) @ Q
            X2 = A2.to(device=device, dtype=torch.float32) @ Q
            X3 = A3.to(device=device, dtype=torch.float32) @ Q
            X = torch.cat([X1, X2, X3], dim=0)
            del X1, X2, X3, Q
            torch.cuda.empty_cache()

            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

        n = A1.shape[0]
        Q1 = Q[:n]
        Q2 = Q[n:2*n]
        Q3 = Q[2*n:3*n]

        B = Q1.T @ A1.to(device=device, dtype=torch.float32)
        B += Q2.T @ A2.to(device=device, dtype=torch.float32)
        B += Q3.T @ A3.to(device=device, dtype=torch.float32)

        Ub, S_joint, Vh = torch.linalg.svd(B, full_matrices=False)
        V_joint = Vh.mH
        del B, Vh, Q1, Q2, Q3
        torch.cuda.empty_cache()

        U_joint = Q.matmul(Ub)
        del Q, Ub
        torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)
    Y = V_joint @ S_sqrt_mat

    Delta1 = U2 - U1
    Delta2 = U3 - U1
    X_new = U3 + beta * Delta1 - alpha * Delta2

    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed


def joint_svd_unlearning_v7_binary_target0(R_original, R_ideal, R_pred,
                             w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
                             alpha: float = 0.5, beta: float = 0.5,
                             num_iter: int = 20, num_dim: int = 48,
                             eps: float = 1e-8):
    # Masks
    positive_mask = (R_original == 1)
    unobserved_mask = (R_original == 0)
    unlearn_mask = (R_original - R_ideal) != 0

    n = R_original.shape[0]
    ones = torch.ones((n, 1), device=R_pred.device, dtype=R_pred.dtype)
    zeros = torch.zeros((n, 1), device=R_pred.device, dtype=R_pred.dtype)

    R_original_adjusted = torch.where(positive_mask, ones, zeros)

    positive_mask_ideal = (R_ideal == 1)
    R_ideal_adjusted = torch.where(positive_mask_ideal, ones, torch.where(unlearn_mask, zeros, zeros))

    try:
        torch.cuda.empty_cache()
        R_weighted = torch.cat([R_original_adjusted, R_ideal_adjusted, R_pred], dim=0).to(device='cuda', dtype=torch.float32)
        U_joint, S_joint, V_joint = torch.svd_lowrank(R_weighted, q=num_dim, niter=num_iter)
        del R_weighted
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A1 = R_original_adjusted
        A2 = R_ideal_adjusted
        A3 = R_pred

        ncols = A1.shape[1]
        Omega = torch.randn(ncols, num_dim, device=device, dtype=torch.float32)

        Y1 = A1.to(device=device, dtype=torch.float32) @ Omega
        Y2 = A2.to(device=device, dtype=torch.float32) @ Omega
        Y3 = A3.to(device=device, dtype=torch.float32) @ Omega
        X = torch.cat([Y1, Y2, Y3], dim=0)
        del Y1, Y2, Y3
        torch.cuda.empty_cache()

        Q = torch.linalg.qr(X).Q

        for _ in range(num_iter):
            n_users = A1.shape[0]
            Q1 = Q[:n_users]
            Q2 = Q[n_users:2*n_users]
            Q3 = Q[2*n:3*n]

            X = A1.to(device=device, dtype=torch.float32).T @ Q1
            X += A2.to(device=device, dtype=torch.float32).T @ Q2
            X += A3.to(device=device, dtype=torch.float32).T @ Q3
            del Q1, Q2, Q3
            torch.cuda.empty_cache()
            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

            X1 = A1.to(device=device, dtype=torch.float32) @ Q
            X2 = A2.to(device=device, dtype=torch.float32) @ Q
            X3 = A3.to(device=device, dtype=torch.float32) @ Q
            X = torch.cat([X1, X2, X3], dim=0)
            del X1, X2, X3, Q
            torch.cuda.empty_cache()

            Q = torch.linalg.qr(X).Q
            del X
            torch.cuda.empty_cache()

        n = A1.shape[0]
        Q1 = Q[:n]
        Q2 = Q[n:2*n]
        Q3 = Q[2*n:3*n]

        B = Q1.T @ A1.to(device=device, dtype=torch.float32)
        B += Q2.T @ A2.to(device=device, dtype=torch.float32)
        B += Q3.T @ A3.to(device=device, dtype=torch.float32)

        Ub, S_joint, Vh = torch.linalg.svd(B, full_matrices=False)
        V_joint = Vh.mH
        del B, Vh, Q1, Q2, Q3
        torch.cuda.empty_cache()

        U_joint = Q.matmul(Ub)
        del Q, Ub
        torch.cuda.empty_cache()

    n = R_original.shape[0]
    U1 = U_joint[:n]
    U2 = U_joint[n:2*n]
    U3 = U_joint[2*n:3*n]

    sqrt_S = torch.sqrt(S_joint)
    S_sqrt_mat = torch.diag(sqrt_S)
    Y = V_joint @ S_sqrt_mat

    Delta1 = U2 - U1
    Delta2 = U3 - U1
    X_new = U3 + beta * Delta1 - alpha * Delta2

    del U_joint, S_joint, V_joint, U1, U2, U3, sqrt_S, Delta1
    torch.cuda.empty_cache()

    user_embed = X_new  @ S_sqrt_mat
    item_embed = Y

    return user_embed, item_embed