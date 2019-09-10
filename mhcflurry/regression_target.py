import numpy


def from_ic50(ic50, max_ic50=50000.0):
    """
    Convert ic50s to regression targets in the range [0.0, 1.0].
    
    Parameters
    ----------
    ic50 : numpy.array of float

    Returns
    -------
    numpy.array of float

    """
    x = 1.0 - (numpy.log(numpy.maximum(ic50, 1e-12)) / numpy.log(max_ic50))
    return numpy.minimum(
        1.0,
        numpy.maximum(0.0, x))


def to_ic50(x, max_ic50=50000.0):
    """
    Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [0, 50000.0].
    
    Parameters
    ----------
    x : numpy.array of float

    Returns
    -------
    numpy.array of float
    """
    return max_ic50 ** (1.0 - x)
