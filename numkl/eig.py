from numkl.ev import syevd


def eighx(a):
    return syevd(a, matrix_layout=1, jobz="V")


def eigvalshx(a):
    return syevd(a, matrix_layout=1, jobz="N")
