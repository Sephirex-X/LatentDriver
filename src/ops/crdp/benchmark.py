import time
import rdp
import crdp
import numpy as np

def test(f, arr, epsilon):
    start = time.time()
    res = f(arr, epsilon)
    duration = time.time() - start
    print(f'{f.__module__} {len(arr)} points, epsilon={epsilon}, duration {duration * 1000} ms')
    return res

np.random.seed(0)
for data_size in [100, 1000, 10000]:
    for epsilon in [0, 0.01, 0.1, 0.5, 1]:
        arr = np.random.rand(data_size, 2)
        a = test(rdp.rdp, arr, epsilon)
        b = test(crdp.rdp, arr, epsilon)
        assert np.all(a == b)