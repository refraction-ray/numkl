errmsg = {
    "wrongtype": "Unsuported numerical type for lapack routine",
    "workmemory": "Failed to allocate memory for a working array",
    "transposememory": "Failed to allocate memory for transposed matrix",
}


def _info_error(info):
    if info > 0:
        raise ValueError("MKL Internal error")
    elif info == -1010:
        raise MemoryError(errmsg["workmemory"])
    elif info == -1011:
        raise MemoryError(errmsg["transposememory"])
    else:
        raise ValueError("The %s input parameters is illegal" % (-info))
