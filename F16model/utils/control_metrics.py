import numpy as np


def relative_error(y, y_approx):
    """
    y : some value y
    y_approx : approximation of y
    """
    assert len(y) == len(y_approx), "Arrys have different size"
    arr_size = len(y)
    return (1 / arr_size) * np.sum(
        np.abs(y - y_approx) / (np.abs(y) + 1e-10)
    )  # this metric total trash if you have 0 in data set


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


def nMAE(y_true, predictions):
    mae_value = mae(y_true, predictions)
    nmae = mae_value / np.mean(np.abs(y_true) + 1e-9)
    return nmae
