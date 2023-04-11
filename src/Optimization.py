import utility as util
from cluster import *
import query as query

def sway1(data):
    """
    Function:
        sway1
    Description:
        Finds the best half of the data by recursion (Menzies)
    Input:
        data - data to sway
    Output:
        Swayed data
    """
    def worker(rows, worse, evals0, above = None):
        if len(rows) <= len(data.rows) ** util.args.min:
            return rows, many(worse, util.args.rest*len(rows)), evals0
        else:
            l , r, A, B, c, evals = half(data, rows, None, above)
            if query.better(data, B, A):
                l, r, A, B = r, l, B, A
            for row in r:
                worse.append(row)
            return worker(l, worse, evals + evals0, A)
    best, rest, evals = worker(data.rows, [], 0)
    return DATA(data, best), DATA(data, rest), evals

def sway2(data):
    """
    Function:
        sway2
    Description:
        Finds the best half of the data by recursion (Group 8 version)
    Input:
        data - data to sway
    Output:
        Swayed data
    """
    def worker(rows, worse, evals0, above = None):
        if len(rows) <= len(data.rows) ** util.args.min:
            return rows, many(worse, util.args.rest*len(rows)), evals0
        else:
            l, r, A, B, c, evals = half(data, rows, None, above, True)
            l_avg = []
            r_avg = []
            for dataHalf in [l, r]:
                symDict = {}
                for index in range(len(dataHalf[0])):
                    if isinstance(data.cols.all[index].col, NUM):
                        colSum = 0
                        for inner_list in dataHalf:
                            if inner_list[index] == "?":
                                # Use average of the col if "?"
                                colSum += (sum(data.cols.all[index].col.has) /
                                           data.cols.all[index].col.n) if data.cols.all[index].col.n != 0 else 0
                            else:
                                colSum += inner_list[index]
                        if dataHalf == l:
                            l_avg.append(colSum / len(dataHalf))
                        else:
                            r_avg.append(colSum / len(dataHalf))
                    else:
                        for inner_list in dataHalf:
                            if not inner_list[index] in symDict:
                                symDict[inner_list[index]] = 1
                            else:
                                symDict[inner_list[index]] += 1
                        if dataHalf == l:
                            l_avg.append(max(symDict, key=symDict.get))
                        else:
                            r_avg.append(max(symDict, key=symDict.get))
            if query.better(data, r_avg, l_avg):
                l, r, A, B = r, l, B, A
            for row in r:
                worse.append(row)
            return worker(l, worse, evals + evals0, A)
    best, rest, evals = worker(data.rows, [], 0)
    return DATA(data, best), DATA(data, rest), evals
