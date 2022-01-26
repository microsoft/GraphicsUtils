# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np


def sigmoid(x):
    pos_x_mask = x >= 0
    neg_x_mask = x < 0
    z = np.zeros_like(x)
    z[pos_x_mask] = np.exp(-x[pos_x_mask])
    z[neg_x_mask] = np.exp(x[neg_x_mask])
    t = np.ones_like(x)
    t[neg_x_mask] = z[neg_x_mask]
    return t / (1 + z)


def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
    xyz_points_pad = np.pad(points_2d, ((0, 1), (0, 1), (0, 0)), mode='symmetric')
    xyz_points_ver = (xyz_points_pad[:, :-1, :] - xyz_points_pad[:, 1:, :])[:-1, :, :]
    xyz_points_hor = (xyz_points_pad[:-1, :, :] - xyz_points_pad[1:, :, :])[:, :-1, :]
    xyz_normal = np.cross(xyz_points_hor, xyz_points_ver)
    xyz_dist = np.linalg.norm(xyz_normal, axis=-1, keepdims=True)
    xyz_normal = np.divide(xyz_normal, xyz_dist, out=np.zeros_like(xyz_normal), where=xyz_dist != 0)
    return xyz_normal
