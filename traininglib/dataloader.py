import multiprocessing
import multiprocessing.pool
import os
import typing as tp

import numpy as np



class ThreadedDataLoader:
    '''Like pytorch DataLoader but with threads instead of processes
       to avoid crashes. '''
    def __init__(
        self, 
        dataset:    tp.Sequence, 
        batch_size: int,
        collate_fn: tp.Optional[tp.Callable] = None,
        n_workers:  tp.Optional[int] = None,
        shuffle:    bool = False,
        timeout:    int  = 10,
    ):
        assert (
            batch_size > 0 and type(batch_size) == int
        ), "batch_size must be a positive integer"

        self.dataset    = dataset
        self.batch_size = batch_size
        self.n_workers  = n_workers
        self.shuffle    = shuffle
        self.timeout    = timeout
        self.collate_fn = collate_fn


    def __len__(self) -> int:
        """Number of batches per epoch"""
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        batched_indices = [
            indices[i:][: self.batch_size].tolist()
                for i in range(0, len(indices), self.batch_size)
        ]

        with multiprocessing.pool.ThreadPool(self.n_workers) as pool:
            for batch_of_indices in batched_indices:
                batch = pool.map(lambda i: self.dataset[i], batch_of_indices)
                if self.collate_fn is not None:
                    batch = self.collate_fn(batch)
                yield batch



