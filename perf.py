# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import abstractmethod
import multiprocessing

from typing import List, Tuple, Dict


class MultiInstance:
    """
    The abstract class
    """
    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)

    @abstractmethod
    def call(self, worker_id, offset, samples, *args, **kwds):
        """
        Start the multiple processing
        """


def multiple_processor(func: MultiInstance, samples: List, workers, args: Tuple,
    kargs: Dict, split_tasks=False):
    """
    Start multiple processing
    """
    if split_tasks:
        assert len(samples) == workers

    samples_per_worker = int((len(samples) - 1) / workers + 1)
    processes = list()
    for worker in range(workers):
        if split_tasks:
            start_index = 0
            split_samples = samples[worker]
        else:
            start_index = worker * samples_per_worker
            end_index = min((worker + 1) * samples_per_worker, len(samples))
            split_samples = samples[start_index: end_index]
        f_args = (worker, start_index, split_samples) + args
        f_target = func()
        thread = multiprocessing.Process(target=f_target, args=f_args, kwargs=kargs)
        processes.append(thread)
        thread.start()
    for process in processes:
        process.join()
