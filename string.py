# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import pickle
import inspect

from typing import Dict, List

import numpy as np


def reorganize_by_symbol(tar: str, symbol: str, sliced: slice) -> str:
    return symbol.join(tar.split(symbol)[sliced])


def abs_dir_path(tar: str):
    return os.path.dirname(os.path.abspath(tar))


def mkdir_automated(tar: str):
    os.makedirs(tar, mode=0o775, exist_ok=True)
    return tar


def pickle_serialization(file_name: str, op_type: str, obj: object=None):
    with open(file_name, op_type) as fp:
        if op_type == 'rb':
            return pickle.loads(fp.read())
        elif op_type == 'wb':
            pickle.dump(obj, fp)
            return list()
        else:
            raise NotImplementedError


def match_function_dict(param_dict: Dict, external_dict: Dict, target_func: object):
    for func_key in inspect.signature(target_func).parameters.keys():
        if func_key not in external_dict.keys():
            continue
        param_dict[func_key] = external_dict[func_key]
    return param_dict


def add_variant_length_array(a: np.ndarray, b: np.ndarray):
    if a.shape[0] > b.shape[0]:
        tmp = a
        tmp[:b.shape[0]] = tmp[:b.shape[0]] + b
    else:
        tmp = b
        tmp[:a.shape[0]] = tmp[:a.shape[0]] + a
    return tmp


class StackList(object):
    def __init__(self):
        self.stacked_list = list()

    def append(self, inputs: List):
        if self.stacked_list:
            for i, s in zip(inputs, self.stacked_list):
                s.append(i)
        else:
            for i in inputs:
                self.stacked_list.append([i])

    def extend(self, inputs: List):
        if self.stacked_list:
            for i, s in zip(inputs, self.stacked_list):
                s.extend(i)
        else:
            for i in inputs:
                self.stacked_list.append(i)


class Accumulator(object):
    def __init__(self):
        self._accumulate = list()

    def acc(self, value):
        self._accumulate.append(value)

    def acc_ext(self, value):
        self._accumulate.extend(value)

    def mean(self):
        return np.array(self._accumulate).mean()

    def mediate(self):
        return np.sort(self._accumulate)[int(len(self._accumulate) / 2)]

    def ratio_threshold(self, func):
        base_acc = np.array(self._accumulate)
        return np.count_nonzero(base_acc[func(base_acc)]) / base_acc.shape[0] * 100

    def reset(self):
        self._accumulate = list()


class ArrayAccumulator(object):
    def __init__(self):
        self._accumulate = StackList()

    @property
    def accumulate(self):
        return [x for x in self._accumulate.stacked_list]

    def acc(self, value):
        self._accumulate.append(value)

    def acc_ext(self, value):
        self._accumulate.extend(value)

    def mean(self):
        return [np.array(x).mean() if len(x) > 1 else x for x in self._accumulate.stacked_list]

    def mediate(self):
        return [np.sort(x)[int(len(x) / 2)] for x in self._accumulate.stacked_list]

    def ratio_threshold(self, threshold):
        return [np.count_nonzero(np.array(x)[np.array(x) < threshold]) / len(x) for x in self._accumulate.stacked_list]

    def reset(self):
        self._accumulate = StackList()


class LogParser(object):
    def __init__(self, log_path: str):
        self._content = dict(epoch=list(), vis=list(), all=list(), loss=list())
        with open(log_path, 'r') as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                epoch_start = line.find('Train Epoch')
                if epoch_start == -1:
                    continue
                line_info = line[epoch_start:-1].split('-')
                epoch_id = int(line_info[0].rstrip().split(' ')[-1])
                epoch_loss = float(line_info[-1].rstrip().split(':')[-1])
                vis_acc = np.fromstring(fp.readline().split('VIS')[-1], dtype=np.float32, sep=' ')
                all_acc = np.fromstring(fp.readline().split('ALL')[-1], dtype=np.float32, sep=' ')
                self._content['epoch'].append(epoch_id)
                self._content['vis'].append(vis_acc)
                self._content['all'].append(all_acc)
                self._content['loss'].append(epoch_loss)

    def select_top_n_by_category(self, top_n, category, item):
        target_values = np.array(self._content[category])[..., item]
        top_n_index = np.argsort(target_values)[::-1][:top_n]
        target_epoch = np.array(self._content['epoch'])[top_n_index]
        return target_epoch
