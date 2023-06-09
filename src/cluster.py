from query import *
from data import *
import utility as util



def half(data, rows = None, cols = None, above = None, useDist2 = False):
        """
        Function:
            half
        Description:
            Splits data in half
        Input:
            self - current DATA instance
            rows - rows to split
            cols - cols to split
            above - previous point of split
        Output:
            left - list of rows to the left of split
            right - list of rows to the right of split
            A - far left point
            B - far right point
            mid - mid point where split occurs
            c - Distance between A and B
        """
        def gap(r1, r2):
            if (not useDist2):
                return dist(data, r1, r2, cols)
            return dist2(data, r1, r2, cols)
        def cos(a, b, c):
            return (a**2 + c**2 - b**2)/((2 * c) + 1E-32)
        def proj(r):
            return {'row': r, 'x': cos(gap(r, A), gap(r, B), c)}
        rows = rows or data.rows
        some = many(rows, util.args.Halves)
        A = above if util.args.Reuse else any(some)
        tmp = sorted([{"row": r, "d": gap(r, A)} for r in some], key=lambda x: x["d"])
        far = tmp[int(len(tmp)*util.args.Far)]
        B, c = far["row"], far["d"]
        sorted_rows = sorted(map(proj, rows), key=lambda x: x["x"])
        left, right = [], []
        for n, two in enumerate(sorted_rows):
            if n <= (len(rows) - 1) / 2:
                left.append(two["row"])
            else:
                right.append(two["row"])
        evals = 1 if (hasattr(util.args, "Reuse") and above) else 2
        return left, right, A, B, c, evals

def tree(data, rows = None, cols = None, above = None):
    rows = rows if rows else data.rows
    here = {"data" : DATA(data, rows)}
    if len(rows)>=2*(len(data.rows) ** util.args.min):  # util.args.min needed where 0.5 is
        left, right, A, B, _, _ = half(data, rows, cols, above)
        here["left"] = tree(data, left, cols, A)
        here["right"] = tree(data, right, cols, B)
    return here

def showTree(tree, lvl=0):
    """
    Function:
        showTree
    Description:
        Displays optimization of data as a tree
    Input:
        tree - data
        lvl - how deep the tree is
    Output:
        None
    """
    if tree:
        print("{}[{}]".format("|.. " * lvl, len(tree["data"].rows)), end="")
        if lvl == 0 or not "left" in tree:
            print(stats(tree["data"]))
        else:
            print("")
        showTree(tree["left"] if "left" in tree else None, lvl + 1)
        showTree(tree["right"] if "right" in tree else None, lvl + 1)
