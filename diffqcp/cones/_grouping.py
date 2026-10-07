"""Helpers for grouping consecutive same-sized cones (they are projected with one `vmap`)."""


def _group_cones_in_order(dims: list[int] | list[float]) -> list[list[int | float]]:
    """Group consecutive same-sized cones while preserving order.

    For a list of cone dimensions (so for a specific cone), returns a

    """
    if not isinstance(dims, list):
        raise ValueError(f"`dims` must be a `list`, but a {type(dims)} was provided.")

    groups: list[list[int | float]] = [[dims[0]]]
    for d in dims[1:]:
        if d == groups[-1][-1]:
            groups[-1].append(d)
        else:
            groups.append([d])

    return groups


def _collect_cone_batch_info(groups: list[list[int | float]]) -> list[tuple[int, int]]:
    """
    Returns a list of tuples such that for the ith group in groups,
    the 0th element in the ith tuple is the dimension of the cone for the ith
    group and the 1st element in the tuple is the number of those
    """
    dims_batches = []
    for group in groups:
        dims_batches.append((int(group[0]), len(group)))
    return dims_batches
