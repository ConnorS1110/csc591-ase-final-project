from col import COL
from range import *
import update as update
from query import *
import utility as util
import math
from copy import deepcopy
import list as lst
from rule import RULE

def bins(cols, rowss):
    """
    Function:
        bins
    Description:
        Computes the bins (ranges of values) for a list of
        columns based on the values of each column in a list of rows.

    Input:
        cols:
            a list of columns to compute the bins for.
        rowss:
            a dictionary of rows where each key represents a row label and the value is a list of values for each column.
    Output:
        out
            a list of computed bins (ranges of values) for each column.
            Each bin is represented as a list of Range objects. If the column is symbolic,
            the list will only contain Range objects. If the column is not symbolic,
            the list may contain merged ranges of adjacent bins.
    """
    out = []
    for col in cols:
        ranges = {}
        for y, rows in rowss.items():
            for row in rows:
                if (isinstance(col, COL)):
                    col = col.col
                x = row[col.at]
                if x != "?":
                    k = int(bin(col, float(x) if x != "?" else x))
                    ranges[k] = ranges[k] if k in ranges else RANGE(col.at, col.txt, float(x) if x != "?" else x)
                    update.extend(ranges[k], float(x), y)
        ranges = {key: value for key, value in sorted(ranges.items(), key=lambda x: x[1].lo)}
        newRanges = {}
        i = 0
        for key in ranges:
            newRanges[i] = ranges[key]
            i += 1
        newRangesList = []
        if hasattr(col, "isSym") and col.isSym:
            for item in newRanges.values():
                newRangesList.append(item)
        out.append(newRangesList if hasattr(col, "isSym") and col.isSym else mergeAny(newRanges))
    return out

def bin(col, x):
    """
    Function:
        bin

    Description:
        The bin function takes a column object col and a value x as
        input and returns the corresponding bin value for x based on
        the range of col. If x is "?" or col is a symbol column, then
        the function simply returns x.

    Input:
        col - A column object containing the range of values to be binned.
        x - A value to be binned.

    Output:
        The corresponding bin value for x based on the range of col.
    """
    if x=="?" or hasattr(col, "isSym"):
        return x
    tmp = (col.hi - col.lo)/(util.args.bins - 1)
    return 1 if col.hi == col.lo else math.floor(x / tmp + 0.5) * tmp

def mergeAny(ranges0):
    """
    Function:
        mergeAny

    Description:
        The mergeAny function takes a list of range objects ranges0
        as input and recursively merges adjacent ranges until there
        are no more adjacent ranges to merge. The resulting ranges
        are returned in a list.

    Input:
        ranges0 - A list of range objects.

    Output:
        A list of range objects resulting from merging adjacent ranges in ranges0.
    """
    def noGaps(t):
        for j in range(1, len(t)):
            t[j].lo = t[j-1].hi
        t[0].lo = -float("inf")
        t[-1].hi = float("inf")
        return t
    if isinstance(ranges0, dict): ranges0 = list(ranges0.values())
    ranges1, j = [], 0
    while j < len(ranges0):
        left, right = ranges0[j], ranges0[j+1] if j + 1 < len(ranges0) else None
        if right:
            y = merge2(left.y, right.y)
            if y:
               j = j+1
               left.hi, left.y = right.hi, y
        ranges1.append(left)
        j += 1
    return noGaps(ranges0) if len(ranges1)==len(ranges0) else mergeAny(ranges1)

def merge2(col1, col2):
    """
    Function:
        merge2

    Description:
        The merge2 function takes two columns col1 and col2 as inputs,
        merges them using the merge function, and returns the merged
        column if the distance between the merged column and the individual
        columns is less than or equal to the expected distance based on their sizes.

    Input:
    col1 - A column object containing data to be merged.
    col2 - A column object containing data to be merged with col1.

    Output:
        The merged column if the distance between the merged column
        and the individual columns is less than or equal to the expected
        distance based on their sizes. Otherwise, no output is returned.
    """
    new = merge(col1, col2)
    if div(new) <= (div(col1)*col1.n + div(col2)*col2.n)/new.n:
        return new

def merge(col1, col2):
    """
    Function:
        merge

    Description:
        The merge function takes two columns col1 and col2 as inputs and returns
        a new column that contains all the data from both input columns. If col1
        has the isSym attribute set to True, it merges the data in col2 with the
        data in col1 using the has dictionary. Otherwise, it merges the data using
        the has list. Additionally, if col1 does not have the isSym attribute set to True,
        it updates the lo and hi attributes of the new column to the minimum and maximum
        values of the lo and hi attributes of the input columns, respectively.

    Input:
        col1 - A column object containing data to be merged.
        col2 - A column object containing data to be merged with col1.

    Output:
        A new column object that contains all the data from col1 and col2.
    """
    new = deepcopy(col1)
    if hasattr(col1, "isSym") and col1.isSym:
        for x, n in col2.has.items():
            add(new, x, n)
    else:
        for n in col2.has:
            add(new, n)
        new.lo = min(col1.lo, col2.lo)
        new.hi = max(col1.hi, col2.hi)
    return new

def xpln1(data, best, rest, doPrint = True):
    """
    Function:
        xpln1
    Description:
        Finds a rule explanation of our optimized best/rest data (Menzies)
    Input:
        data - data to explain
        best- best set of data
        rest - all swayed data not in best
    Output:
        Rule explanation
    """
    def v(has):
        return value(has, len(best.rows), len(rest.rows), "best")
    def score(ranges):
        rule = RULE(ranges, maxSizes)
        if rule:
            if doPrint: print(showRule(rule))
            bestr= selects(rule, best.rows)
            restr= selects(rule, rest.rows)
            if len(bestr) + len(restr) > 0:
                return v({"best": len(bestr), "rest": len(restr)}), rule
    tmp, maxSizes = [], {}
    for ranges in bins(data.cols.x, {"best": best.rows, "rest": rest.rows}):
        maxSizes[ranges[0].txt] = len(ranges)
        if doPrint: print("")
        for range in ranges:
            if doPrint: print(range.txt, range.lo, range.hi)
            tmp.append({"range": range, "max": len(ranges), "val": v(range.y.has)})

    rule, most = firstN(sorted(tmp, key=lambda x: x["val"], reverse=True), score, doPrint)
    return rule, most

def firstN(sortedRanges, scoreFun, doPrint = True):
    if doPrint:
        print("")
        for r in sortedRanges:
            print(r["range"].txt, r["range"].lo, r["range"].hi, round(r["val"], 2), r["range"].y.has)
    first = sortedRanges[0]["val"]
    def useful(range):
        if range["val"] > 0.05 and range["val"] > first / 10:
            return range

    sortedRanges = list(filter(useful, sortedRanges))
    most, out = -1, None

    for n in range(len(sortedRanges)):
        tmp, rule = scoreFun([r["range"] for r in sortedRanges[:n + 1]]) or (None, None)

        if tmp and tmp > most:
            out, most = rule, tmp

    return out, most

def showRule(rule):
    def pretty(range):
        return range['lo'] if range['lo'] == range['hi'] else [range['lo'], range['hi']]

    def merges(attr, ranges):
        return list(map(pretty, merge(sorted(ranges, key=lambda r: r['lo'])))), attr

    def merge(t0):
        t, j = [], 0
        while j < len(t0):
            left, right = t0[j], t0[j+1] if j+1 < len(t0) else None
            if right and left['hi'] == right['lo']:
                left['hi'] = right['hi']
                j += 1
            t.append({'lo': left['lo'], 'hi': left['hi']})
            j += 1
        return t if len(t0) == len(t) else merge(t)

    return lst.kap(rule, merges)

def selects(rule, rows):
    def disjunction(ranges, row):
        for range in ranges:
            lo = int(range['lo']) if isinstance(range['lo'], str) else range['lo']
            hi = int(range['hi']) if isinstance(range['hi'], str) else range['hi']
            at = int(range['at'])
            x = row[at]
            if x == "?":
                return True
            x = float(x)
            if lo == hi and lo == x:
                return True
            if lo <= x and x < hi:
                return True
        return False

    def conjunction(row):
        for ranges in rule.values():
            if not disjunction(ranges, row):
                return False
        return True

    return [r for r in rows if conjunction(r)]
