# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np


def mesh_grid_nd(vox_size):
    dim_index = [np.arange(a, dtype=np.int32) for a in vox_size]
    dim_index = np.stack(np.meshgrid(*dim_index, indexing='ij'), axis=-1)
    return dim_index
