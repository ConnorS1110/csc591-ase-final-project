from utility import *
import math
import utility as util
import list as listModule

Seed = 937162211

def itself(x):
    return x

def cliffsDelta(ns1, ns2):
    """
    Function:
        cliffsDelta
    Description:
        Calculates Cliff's delta effect size measure
    Input:
        ns1 - first list of rows
        ns2 - second list of rows
    Output:
        Cliff's delta effect size
    """
    if len(ns1) > 256: ns1 = listModule.many(ns1, 256)
    if len(ns2) > 256: ns2 = listModule.many(ns2, 256)
    if len(ns1) > (10 * len(ns2)): ns1 = listModule.many(ns1, 10 * len(ns2))
    if len(ns2) > (10 * len(ns1)): ns2 = listModule.many(ns2, 10 * len(ns1))
    n, gt, lt = 0, 0, 0
    for x in ns1:
        for y in ns2:
            n += 1
            if x > y: gt += 1
            if x < y: lt += 1
    return abs(lt - gt) / n > util.args.cliffs

def diffs(nums1, nums2):
    """
    Function:
        diffs
    Description:
        Creates a list of cliff's deltas and a textual description of the col
    Input:
        nums1 - first list of rows
        nums2 - second list of rows
    Output:
        List of cliff's deltas and a textual description of the col
    """
    def kap(nums, fn):
        return [fn(k, v) for k, v in enumerate(nums)]
    return kap(nums1, lambda k, nums: (cliffsDelta(nums.col.has, nums2[k].col.has), nums.col.txt))
