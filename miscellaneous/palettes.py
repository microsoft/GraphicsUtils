# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np


D3C20_RGB = None


def d3c20_hex():
    d3c20_coding = np.asarray(
        [0xffffff,
         0x1f77b4, 0xaec7e8, 0xff7f0e, 0xffbb78, 0x2ca02c, 0x98df8a, 0xd62728, 0xff9896, 0x9467bd, 0xc5b0d5,
         0x8c564b, 0xc49c94, 0xe377c2, 0xf7b6d2, 0x7f7f7f, 0xc7c7c7, 0xbcbd22, 0xdbdb8d, 0x17becf, 0x9edae5,
         0x42bc66, 0xc83683, 0x4e47b7, 0x8c39c5, 0x000000],
        dtype=np.uint32)
    return d3c20_coding


def d3c20_rgb():
    global D3C20_RGB
    if D3C20_RGB is not None:
        return D3C20_RGB
    hex_coding = d3c20_hex()
    hex_b = np.bitwise_and(hex_coding, 0xff)
    hex_g = np.bitwise_and(np.right_shift(hex_coding, 8), 0xff)
    hex_r = np.bitwise_and(np.right_shift(hex_coding, 16), 0xff)
    D3C20_RGB = np.stack([hex_r, hex_g, hex_b], axis=-1).astype(np.uint8)
    return D3C20_RGB
