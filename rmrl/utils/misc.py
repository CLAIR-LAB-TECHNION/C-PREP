import hashlib
from itertools import chain, combinations


def powerset(iterable, max_subset_len=None):
    # have iterable as a list
    lst = list(iterable)

    if max_subset_len is None:
        max_subset_len = len(lst) + 1

    return chain.from_iterable(combinations(lst, r) for r in range(max_subset_len))


def split_pairs(lst):
    return [
        (lst[i], lst[i + 1])
        for i in range(0, len(lst), 2)
    ]


def sha3_hash(item):
    return hashlib.sha3_256(str(item).encode()).hexdigest()
