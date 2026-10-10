"""Helpers for grouping consecutive same-sized cones (they are projected with one `vmap`)."""


def _group_cones_in_order(dims: list[int] | list[float]) -> list[list[int | float]]:
    """Group consecutive same-sized cones while preserving order.

    For example, `[3, 3, 4, 3]` becomes `[[3, 3], [4], [3]]`. Only *consecutive*
    equal entries are grouped, so the cone vector's block order is unchanged.

    **Arguments:**
    - `dims`: the sizes of the cones of one type, in the order they appear in the
        cone vector. Must be a non-empty `list`.

    **Returns:**

    A list of groups, each a list of equal consecutive entries of `dims`.
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
    """Summarize each group from `_group_cones_in_order` as `(size, count)`.

    **Arguments:**
    - `groups`: the output of `_group_cones_in_order`.

    **Returns:**

    A list with one tuple per group: the (integer) cone size shared by the group
    and the number of cones in it.
    """
    dims_batches = []
    for group in groups:
        dims_batches.append((int(group[0]), len(group)))
    return dims_batches
