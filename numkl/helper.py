errmsg = {"wrongtype": "Unsuported numerical type for lapack routine"}


def _info_error(info):
    if info > 0:
        raise ValueError("MKL Internal error")
    else:
        raise ValueError("The %s input parameters is illegal" % (-info))
