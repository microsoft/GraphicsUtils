# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import multiprocessing

from typing import List, Tuple


def multiple_processor(func, samples: List, workers, args: Tuple):
    samples_per_worker = int((len(samples) - 1) / workers + 1)
    processes = list()
    for w in range(workers):
        start_index = w * samples_per_worker
        end_index = min((w + 1) * samples_per_worker, len(samples))
        f_args = (samples[start_index: end_index], ) + args + (start_index, w)
        t = multiprocessing.Process(target=func, args=f_args)
        processes.append(t)
        t.start()
    for p in processes:
        p.join()

