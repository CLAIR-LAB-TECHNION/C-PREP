from itertools import chain, combinations


def powerset(iterable, max_subset_len=None):
    # have iterable as a list
    lst = list(iterable)

    if max_subset_len is None:
        max_subset_len = len(lst) + 1

    return chain.from_iterable(combinations(lst, r) for r in range(max_subset_len))
