"""Robust summation methods for fixed-graph Byzantine-resilient learning."""

import torch


def cs_plus(weights, gradients, byz_weights, **kwargs):
    """Clipped Sampling+ (CS+) robust summation method."""
    distances = gradients.norm(dim=1)

    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - 2*byz_weights

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # nb: weights[idx:] >= byz_weights

    # Compute the clipping threshold
    if idx < 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf

    elif cumulative_weights[idx] == target_weight:
        clipping_threshold = sorted_distances[idx]
    elif cumulative_weights[idx] > target_weight:
        if idx-1 >= 0:
            clipping_threshold = sorted_distances[idx-1]
        else:
            clipping_threshold =  0
    else:
        raise ValueError("Unexpected behavior in computing the adaptive clipping threshold")

    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    num_clipped = mask[:,0].sum().item()
    clipped_differences = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_differences).sum(dim=0), num_clipped


def cs_plus_bis(weights, gradients, byz_weights, **kwargs):
    """CS+ variant that also stores the number of clipped gradients."""
    distances = gradients.norm(dim=1)

    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - 2*byz_weights

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # nb: weights[idx:] >= byz_weights

    # Compute the clipping threshold
    if idx < 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf

    elif cumulative_weights[idx] == target_weight:
        clipping_threshold = sorted_distances[idx]
    elif cumulative_weights[idx] > target_weight:
        if idx-1 >= 0:
            clipping_threshold = sorted_distances[idx-1]
        else:
            clipping_threshold =  0
    else:
        raise ValueError("Unexpected behavior in computing the adaptive clipping threshold")

    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    num_clipped = mask[:,0].sum()
    clipped_differences = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_differences).sum(dim=0), num_clipped


def gts(weights, gradients, byz_weights, **kwargs):
    """Gradient Thresholding with Sampling (GTS) robust summation method."""
    distances = gradients.norm(dim=1)

    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - byz_weights

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # weights(idx:) >= byz_weights

    rest_weight = 0
    if cumulative_weights[idx] > target_weight:
        rest_weight =  cumulative_weights[idx] - target_weight

    sorted_gradients = gradients[sorted_indices,:]
    return (sorted_weights[:idx, None] * sorted_gradients[:idx,:]).sum(dim=0) + sorted_gradients[idx,:] * rest_weight, 0


def cs_he(weights, gradients, byz_weights, **kwargs):
    """Clipped Sampling (He et al. 2022) robust summation method."""
    distances = gradients.norm(dim=1)

    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - byz_weights

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # weights(idx:) >= byz_weights

    # Compute the adaptive clipping threshold of He et al.
    if idx == 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf
    else:
        clipping_threshold = ((sorted_weights[:idx] * sorted_distances[:idx]**2).sum(dim=0) / byz_weights).sqrt().item()

    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    num_clipped = mask[:,0].sum().item()
    clipped_gradients = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_gradients).sum(dim=0), num_clipped
