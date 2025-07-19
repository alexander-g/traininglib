from traininglib.dataloader import ThreadedDataLoader

import threading

import torch


class MockDataset():
    def __init__(self):
        self._called_from_threads = []

    def __getitem__(self, i:int):
        self._called_from_threads.append(threading.current_thread().native_id)
        return torch.rand([3,100,100]), i
    
    def __len__(self):
        return 65

    @staticmethod
    def collate_fn(a):
        x = [x for x,y in a]
        y = [y for x,y in a]
        return torch.stack(x), y



def test_threaded():
    ds = MockDataset()
    ld = ThreadedDataLoader(ds, collate_fn=ds.collate_fn, batch_size=8, n_workers=8, timeout=2, shuffle=True)
    #it = iter(ld)

    
    batches = [batch for batch in ld]
    assert batches[0][0].shape == (8,3,100,100)

    print(ds._called_from_threads)
    assert all([t != threading.current_thread().native_id 
        for t in ds._called_from_threads])
    
    # 2nd round
    batches = [batch for batch in ld]

