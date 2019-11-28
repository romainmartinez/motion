def __split_col_prefix(prefix: str):
    if prefix:
        return lambda x: x.split(prefix)[-1]
    else:
        return lambda x: x
