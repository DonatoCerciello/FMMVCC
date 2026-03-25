import torch
import numpy as np
from tools.augmentations import jitter_torch, scaling_torch, permutation_torch

def generate_pos_neg_index(pseudo_label):
    B = pseudo_label.size(0)
    invalid_index = pseudo_label == -1

    mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(pseudo_label.device)
    mask[invalid_index, :] = False
    mask[:, invalid_index] = False

    mask_eye = torch.eye(B).float().to(pseudo_label.device)
    mask |= mask_eye.bool()

    valid_neg_choices = ~mask.bool().to(pseudo_label.device)

    if valid_neg_choices.sum() == 0:
        neg_indices = torch.randperm(B).to(pseudo_label.device)
        pos_indices= torch.arange(B).to(pseudo_label.device)
        return  pos_indices, neg_indices


    initial_indices = torch.randperm(B).to(pseudo_label.device)
    global_non_cluster_mask = valid_neg_choices.clone().float()

    replace_indices = torch.multinomial(global_non_cluster_mask, num_samples=1, replacement=True).squeeze(1)

    need_replacement = ~valid_neg_choices[range(B), initial_indices]
    initial_indices[need_replacement] = replace_indices[need_replacement]

    neg_indices = initial_indices

    pos_choices = ~valid_neg_choices

    pos_indices = torch.multinomial(pos_choices.float(), num_samples=1, replacement=True).squeeze(1)

    return pos_indices, neg_indices


def MASK(X, missing_rate, num_view=4, important_indices=None, flag=0, alpha=0.5):

    device = X.device
    num_samples, seq_len, feature_dim = X.shape
    H_v = []

    augmentations = [
        "none",
        "jitter",
        "scaling",
        "permutation"
    ]

    for view in range(num_view):

        # Augmentation
        aug_type = augmentations[view % len(augmentations)]
        X_aug = apply_augmentation(X, aug_type)

        view_masks = []

        for sample_idx in range(num_samples):

            sample_important_idx = (
                important_indices[view][sample_idx]
                if (flag != 0 and important_indices is not None)
                else None
            )

            sample_mask = add_mixed_missing_mask(
                seq_len=seq_len,
                feature_dim=feature_dim,
                missing_rate=missing_rate,
                important_idx=sample_important_idx,
                alpha=alpha
            )

            view_masks.append(sample_mask)

        view_masks_tensor = torch.stack(view_masks).to(device)

        # Masking
        view_data = X_aug * view_masks_tensor.float()

        H_v.append(view_data)

    return H_v


def add_mixed_missing_mask(seq_len,
                           feature_dim,
                           missing_rate=0.7,
                           max_continuous_length=5,
                           important_idx=None,
                           alpha=0.5):
    
    # Create two masks: one for non-important features and one for random missingness
    mask_non_important = create_mask(
        seq_len,
        feature_dim,
        missing_rate,
        max_continuous_length,
        important_idx,
        flag=1
    )

    mask_random = create_mask(
        seq_len,
        feature_dim,
        missing_rate,
        max_continuous_length,
        important_idx,
        flag=0
    )

    # Combine the two masks based on alpha
    non_important_zeros = (mask_non_important == 0)
    num_to_set_1 = int((1 - alpha) * non_important_zeros.sum().item())

    if num_to_set_1 > 0:
        zero_indices = np.argwhere(non_important_zeros.numpy())
        selected = np.random.choice(len(zero_indices),
                                    min(num_to_set_1, len(zero_indices)),
                                    replace=False)
        mask_non_important[
            zero_indices[selected][:, 0],
            zero_indices[selected][:, 1]
        ] = True

    random_zeros = (mask_random == 0)
    num_to_set_1 = int(alpha * random_zeros.sum().item())

    if num_to_set_1 > 0:
        zero_indices = np.argwhere(random_zeros.numpy())
        selected = np.random.choice(len(zero_indices),
                                    min(num_to_set_1, len(zero_indices)),
                                    replace=False)
        mask_random[
            zero_indices[selected][:, 0],
            zero_indices[selected][:, 1]
        ] = True

    return mask_non_important | mask_random

def create_mask(seq_len,
                feature_dim,
                missing_rate=0.7,
                max_continuous_length=5,
                important_idx=None,
                flag=0):
    """
    Create boolean mask (True = observed, False = missing)
    """

    total_elements = seq_len * feature_dim
    total_missing = int(total_elements * missing_rate)
    continuous_missing = total_missing // 2
    scattered_missing = total_missing - continuous_missing

    # If important_idx is provided and flag is 1, create a mask that only masks non-important features
    if important_idx is not None and flag == 1:
        mask = torch.zeros((seq_len, feature_dim), dtype=torch.bool)
        mask[important_idx] = 1
        return mask

    mask = torch.ones((seq_len, feature_dim), dtype=torch.bool)
    available_starts = list(range(seq_len))

    # Continuous missing
    while continuous_missing > 0 and available_starts:
        cont_len = min(max_continuous_length,
                       continuous_missing // feature_dim)

        if cont_len <= 0:
            break

        valid_starts = available_starts[:seq_len - cont_len + 1]
        if not valid_starts:
            break

        start = np.random.choice(valid_starts, replace=False)
        mask[start:start + cont_len, :] = False

        for t in range(start, start + cont_len):
            if t in available_starts:
                available_starts.remove(t)

        continuous_missing -= cont_len * feature_dim

    # Scattered missing
    if scattered_missing > 0:
        available_positions = torch.where(mask.flatten())[0].numpy()
        scatter_pos = np.random.choice(
            available_positions,
            min(scattered_missing, len(available_positions)),
            replace=False
        )
        mask_flat = mask.flatten()
        mask_flat[scatter_pos] = False
        mask = mask_flat.reshape(seq_len, feature_dim)

    return mask

def apply_augmentation(x, aug_type):

    if aug_type == "jitter":
        return jitter_torch(x)

    elif aug_type == "scaling":
        return scaling_torch(x)

    elif aug_type == "permutation":
        return permutation_torch(x)

    elif aug_type == "none":
        return x

    else:
        return x