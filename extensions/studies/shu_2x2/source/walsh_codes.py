"""Released Walsh ordering, retained exactly for the two imagery labels."""
import math
import numpy as np


def create_walsh(size_of_walsh=16):
    # Exact released implementation, including row order.
    n = int(math.log(size_of_walsh, 2) + 1)
    j = 1
    W_old = 1
    for i in range(n):
        j_new = 2 ** i
        W_new = np.zeros([j_new, j_new], dtype=np.float32)
        W_new[:j, :j] = W_old
        W_new[j:, :j] = W_old
        W_new[:j, j:] = W_old
        W_new[j:, j:] = abs(1 - W_old)
        W_old = W_new
        j = j_new
    return W_new


def code_rows(n_cls=2):
    if n_cls != 2:
        raise ValueError("The fixed Ma2022 protocol has exactly two imagery classes")
    return create_walsh(16)[:n_cls].astype(np.float32)
