def batch_cone_dims(dims: list[int] | list[float]) -> list[tuple[int, int]]:
    """
    Docstring for `batch_cone_dims`
    
    :param dims: Description
    :type dims: list[int] | list[float]
    :return: Description
    :rtype: list[tuple[int, int]]
    """

    groups = [[dims[0]]]
    dim_batches = []
    for d in dims[1:]:
        if d == groups[-1][-1]:
            groups[-1].append(d)
        else:
            dim_batches.append((groups[-1][0]), len(groups[-1]))
            groups.append([d])

    return dim_batches
