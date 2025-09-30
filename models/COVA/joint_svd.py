import numpy as np
import torch
import math


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

    Delta1 = U1 - U2
    Delta2 = U3 - U1
    X_new = U3 - alpha * Delta1 - beta * Delta2

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