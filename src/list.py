import math
import miscellaneous as misc
import utility as util

def many(t, n):
    """
    Function:
        many
    Description:
        Creates a list of random rows
    Input:
        t - DATA object
        n - Number of row samples
    Output:
        u - list of random n rows from t
    """
    u = []
    for i in range(1, n + 1):
        u.append(any(t))
    return u

def any(t):
    """
    Function:
        any
    Description:
        Selects a random row
    Input:
        t - DATA object
    Output:
        Random row from t
    """
    rintVal = util.rint(None, len(t) - 1)
    return t[rintVal]

def per(t, p):
    """
    Function:
        per
    Description:
        Selects a random row based off a given probability
    Input:
        t - DATA object
        p - probability
    Output:
        Row from t based on probability p
    """
    p = math.floor(((p or 0.5) * len(t)))
    return t[max(0, min(len(t) - 1, p))]

def kap(listOfCols, fun):
    """
    Function:
        kap
    Description:
        Creates map that stores functions as value
    Input:
        listOfCols - list of columns
        fun - anonymous function to be used as value in map
    Output:
        u - map of anonymous functions
    """
    u = {}

    if isinstance(listOfCols, list):
        items = enumerate(listOfCols)
    else:
        items = listOfCols.items()

    for k, v in items:
            v, k = fun(k, v)
            u[k or len(u)+1] = v
    return u

def slice(t, go = None, stop = None, inc = None):
    """
    Function:
        slice
    Description:
        Returns a slice of data
    Input:
        t - data to slice
        go - start point
        stop - stop point
        inc - increment amount
    Output:
        u - Sliced data
    """
    if go and go < 0:
        go = len(t) - 1 + go
    if stop and stop < 0:
        stop = len(t) + stop
    u = []
    for j in range(int((go or 1)), int((stop or len(t))), int(inc or 1)):
        u.append(t[j])
    return u
