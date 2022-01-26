# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np

from skimage import measure


def instance_segmentation_from_semantic(smnt_t, connect=2, denoise_type=None, args=None):
    ff_t, ff_n = measure.label(smnt_t, return_num=True, connectivity=connect)
    ff_n += 1
    smnt_list = np.zeros([ff_n], dtype=np.int32)
    smnt_count = np.zeros([ff_n], dtype=np.int32)
    for f in range(1, ff_n):
        indices = np.transpose(np.argwhere(np.equal(ff_t, f)))
        labels = smnt_t[tuple(indices)]
        label_id = labels[0]
        assert np.all(labels == label_id)
        smnt_list[f] = label_id
        smnt_count[f] = labels.shape[0]
    if denoise_type is None:
        pass
    elif denoise_type == 'regular':
        threshold, = args
        smnt_indices = np.transpose(np.argwhere(smnt_count > threshold))
        smnt_list = smnt_list[tuple(smnt_indices)]
        ff_t = np.where(np.any(ff_t[..., None] == smnt_indices[0], axis=-1), ff_t, np.zeros_like(ff_t))
    else:
        raise NotImplementedError
    return ff_t, smnt_list
