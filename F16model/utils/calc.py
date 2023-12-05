import numpy as np


def normalize_value(value, min_value, max_value, inverse_transform=False):
    """
    Normalize value in range [-1, 1], inverse_transform is denormalization
    """
    low, high = np.float32(min_value), np.float32(max_value)
    if inverse_transform:
        return low + 0.5 * (value + 1.0) * (high - low)
    return 2 * minmaxscaler(value, low, high) - 1


def minmaxscaler(value, min_value, max_value, inverse_transform=False):
    """
    Normalize value in range [0, 1], inverse_transform is denormalization
    """
    if inverse_transform:
        return value * (max_value - min_value) + min_value
    return (value - min_value) / (max_value - min_value)
