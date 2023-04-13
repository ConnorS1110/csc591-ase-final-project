import argparse
import csv
import json
import math
import os
import time
from data import DATA
from num import NUM
from update import *
import query as query
import miscellaneous as misc
import cluster as cluster
import Optimization as opt
import Discretization as disc

help = """
bins: multi-objective semi-supervised discetization
(c) 2023 Tim Menzies <timm@ieee.org> BSD-2

USAGE: lua bins.lua [OPTIONS] [-g ACTIONS]

OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliffs  cliff's delta threshold      = .147
  -d  --d       different is over sd*d       = .35
  -f  --file    data file                    = ../etc/data/auto93.csv
  -F  --Far     distance to distant          = .95
  -g  --go      start-up action              = all
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 4
  -R  --Reuse   child splits reuse a parent pole = false
  -s  --seed    random number seed           = 937162211
"""

args = None
Seed = 937162211
egs = {}
n = 0
smallPositive = 1E-32

def dofile(filename):
    with open(filename) as f:
        return json.load(f)

def rint(lo = None, hi = None):
    """
    Function:
        rint
    Description:
        Makes a random number
    Input:
        low - low value
        high - high value
    Output:
        Random number
    """
    return math.floor(0.5 + rand(lo, hi))

def rand(low = None, high = None):
    """
    Function:
        rand
    Description:
        Creates a random number
    Input:
        low - low value
        high - high value
    Output:
        Random number
    """
    global Seed
    low, high = low or 0, high or 1
    Seed = (16807 * Seed) % 2147483647
    return low + (high - low) * Seed / 2147483647

def eg(key, string, fun):
    """
    Function:
        eg
    Description:
        Creates an example test case and adds it to the dictionary of test cases. Appends the key/value to the actions of the help string
    Input:
        key - key of argument
        string - value of argument as a string
        fun - callback function to use for test case
    Output:
        None
    """
    global egs
    global help
    egs[key] = fun
    help += f"  -g {key}    {string}"

def getCliArgs():
    """
    Function:
        getCliArgs
    Description:
        Parses out the arguments entered or returns an error if incorrect syntax is used
    Input:
        None
    Output:
        None
    """
    global args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-b", "--bins", type=int, default=16, required=False, help="initial number of bins")
    parser.add_argument("-d", "--d", type=float, default=0.35, required=False, help="different is over sd*d")
    parser.add_argument("-g", "--go", type=str, default="all", required=False, help="start-up action")
    parser.add_argument("-h", "--help", action='store_true', help="show help")
    parser.add_argument("-s", "--seed", type=int, default=937162211, required=False, help="random number seed")
    parser.add_argument("-f", "--file", type=str, default="../etc/data/auto93.csv", required=False, help="data file")
    parser.add_argument("-p", "--p", type=int, default=2, required=False, help="distance coefficient")
    parser.add_argument("-c", "--cliffs", type=float, default=0.147, required=False, help="cliff's delta threshold")
    parser.add_argument("-F", "--Far", type=float, default=0.95, required=False, help="distance to distant")
    parser.add_argument("-H", "--Halves", type=int, default=512, required=False, help="search space for clustering")
    parser.add_argument("-m", "--min", type=float, default=0.5, required=False, help="size of smallest cluster")
    parser.add_argument("-M", "--Max", type=int, default=512, required=False, help="numbers")
    parser.add_argument("-r", "--rest", type=int, default=4, required=False, help="how many of rest to sample")
    parser.add_argument("-R", "--Reuse", type=bool, default=False, required=False, help="child splits reuse a parent pole")
    parser.add_argument("--bootstrap", type=int, default = 512, required=False)
    parser.add_argument("--conf", type=float, default = 0.05, required=False)
    parser.add_argument("--cliff", type=float, default = 0.4, required=False)
    parser.add_argument("--cohen", type=float, default = 0.35, required=False)
    parser.add_argument("--Fmt", type=str, default = "%6.2f", required=False)
    parser.add_argument("--width", type=int, default = 40, required=False)

    args = parser.parse_args()

def readCSV(sFilename, fun):
    """
    Function:
        readCSV
    Description:
        reads a CSV and runs a callback function on every line
    Input:
        sFilename - path of CSV file to be read
        fun - callback function to be called for each line in the CSV
    Output:
        None
    """

    # convert numerical values to floats as they appear
    def is_float(s):
        try:
            return float(s)
        except ValueError:
            return s

    with open(sFilename, mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            line = [is_float(value.strip()) for value in line]
            fun(line)

def mean(col):
    """
    Function:
        mean
    Description:
        Returns mean of col
    Input:
        col - col to find mean of
    Output:
        col.mode if col col has isSym and is true, otherwise return the mean value in col
    """
    return col.mode if hasattr(col, "isSym") and col.isSym else sum(col.has) / len(col.has)

def swayFunc():
    """
    Function:
        swayFunc
    Description:
        Callback function to test sway function in DATA class
    Input:
        None
    Output:
        the correct data is output from the sway function
    """
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, args.file)
    data = DATA(full_path)
    best, rest, _ = opt.sway1(data)
    print("\nall ", query.stats(data))
    print("    ",   query.stats(data, query.div))
    print("\nbest", query.stats(best))
    print("    ",   query.stats(best, query.div))
    print("\nrest", query.stats(rest))
    print("    ",   query.stats(rest, query.div))
    print("\nall ~= best?", misc.diffs(best.cols.y, data.cols.y))
    print("best ~= rest?", misc.diffs(best.cols.y, rest.cols.y))

def halfFunc():
    """
    Function:
        halfFunc
    Description:
        Callback function to test half function in DATA class
    Input:
        None
    Output:
        the DATA object is correctly split in half
    """
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, args.file)
    data = DATA(full_path)
    left, right, A, B, c, _ = cluster.half(data)
    print(len(left), len(right))
    l, r = DATA(data, left), DATA(data, right)
    print("l", query.stats(l))
    print("r", query.stats(r))

def binsFunc():
    """
    Function:
        binsFunc
    Description:
        Callback function to test bins function
    Input:
        None
    Output:
        the bins are correctly printed
    """
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, args.file)
    data = DATA(full_path)
    best, rest, _ = opt.sway1(data)
    b4 = None
    print("all","","","", "{best= " + str(len(best.rows)) + ", rest= " + str(len(rest.rows)) + "}")
    result = disc.bins(data.cols.x, {"best": best.rows, "rest": rest.rows})
    for t in result:
        for range in t:
            if range.txt != b4:
                print("")
            b4 = range.txt
            print(range.txt,
                  range.lo,
                  range.hi,
                  round(query.value(range.y.has, len(best.rows), len(rest.rows), "best")),
                  range.y.has)

def explnFunc():
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, args.file)
    data = DATA(full_path)
    best, rest, evals = opt.sway1(data)
    rule, _ = disc.xpln1(data, best, rest)
    print("\n-----------\nexplain=", disc.showRule(rule))
    data1 = DATA(data, disc.selects(rule, data.rows))
    print("all                ", query.stats(data), query.stats(data, query.div))
    print(f"sway with   {evals} evals", query.stats(best), query.stats(best, query.div))
    print(f"xpln on     {evals} evals", query.stats(data1), query.stats(data1, query.div))
    top, _ = query.betters(data, len(best.rows))
    top = DATA(data, top)
    print(f"sort with {len(data.rows)} evals", query.stats(top), query.stats(top, query.div))

def printTables():
    list_of_file_paths = ["../etc/data/auto2.csv", "../etc/data/auto93.csv", "../etc/data/china.csv", "../etc/data/coc1000.csv",
                          "../etc/data/coc10000.csv", "../etc/data/healthCloseIsses12mths0001-hard.csv", "../etc/data/healthCloseIsses12mths0011-easy.csv",
                          "../etc/data/nasa93dem.csv", "../etc/data/pom.csv", "../etc/data/SSM.csv", "../etc/data/SSN.csv"]
    # list_of_file_paths = ["../etc/data/coc1000.csv"]
    script_dir = os.path.dirname(__file__)
    full_file_path = os.path.join(script_dir, "../etc/out/script.out")
    with open(full_file_path, "w") as textFile:
        textFile.write("")
    for file in list_of_file_paths:
        file_string = str(file.split('/')[-1]).split('.')[0]
        print(f"File: {file_string}")
        with open(full_file_path, "a") as textFile:
            textFile.write(f"File: {file_string}\n")
        print("Table 1:")
        table1_dict, table1_xpln_timestamps, xpln_runtimes = table1(file, full_file_path)
        print("Table 2:")
        table2(table1_dict, full_file_path)
        print("Xpln Runtimes:")
        xplnRuntimes(table1_xpln_timestamps, full_file_path, xpln_runtimes)
        print("Sampling Taxes:")
        sampleTaxes(table1_dict, full_file_path)
        print("Explanation Taxes:")
        xplnTaxes(table1_dict, full_file_path)

def table1(filepath, full_file_path):
    global Seed
    ORIGINAL_SEED = Seed - 1
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, filepath)
    data = DATA(full_path)
    row_headers = ["all", opt.sway1, disc.xpln1, opt.sway2, disc.xpln2, "top"]
    max_row_header_len = max(len(s) if isinstance(s, str) else len(s.__name__) for s in row_headers)
    col_headers = []
    for col in data.cols.y:
        col_headers.append(col.col.txt)
    table1_string = f"{'':>30}{''.join(f'{h:>20}' for h in col_headers)}\n"
    table1_current_file_dict = {}
    table1_xpln_timestamps = {}
    clusters = {}
    xpln_runtimes = {}
    for row in row_headers:
        obj_values = []
        table1_current_row_stats = {}
        for i in range(1, 21):
            Seed = ORIGINAL_SEED + i
            data = DATA(full_path)
            # sway1 option
            if (not isinstance(row, str) and row.__name__ == "sway1"):
                row_func = row
                row_string = row.__name__
                best, _, _ = row_func(data)
                data_for_stats = best
            elif (not isinstance(row, str) and row.__name__ == "sway2"):
                row_func = row
                row_string = row.__name__
                best, _, _ = row_func(data)
                data_for_stats = best
            # xpln1 option
            elif (not isinstance(row, str) and row.__name__ == "xpln1"):
                row_func = row
                row_string = row.__name__
                rule = None
                startTime = 0
                while (rule == None):
                    best, rest, _ = opt.sway1(data)
                    startTime = time.time()
                    rule, _ = row_func(data, best, rest, False)
                endTime = time.time()
                data1 = DATA(data, disc.selects(rule, data.rows))
                data_for_stats = data1
                totalTime = endTime - startTime
                if row_string not in table1_xpln_timestamps:
                    table1_xpln_timestamps[row_string] = totalTime
                else:
                    table1_xpln_timestamps[row_string] += totalTime
                if row_string not in xpln_runtimes:
                    xpln_runtimes[row_string] = [totalTime]
                else:
                    xpln_runtimes[row_string].append(totalTime)
            elif (not isinstance(row, str) and row.__name__ == "xpln2"):
                row_func = row
                row_string = row.__name__
                rule = None
                startTime = 0
                while (rule == None):
                    best, rest, _ = opt.sway2(data)
                    startTime = time.time()
                    rule, _ = row_func(data, best, rest, False)
                endTime = time.time()
                data1 = DATA(data, disc.selects(rule, data.rows))
                data_for_stats = data1
                totalTime = endTime - startTime
                if row_string not in table1_xpln_timestamps:
                    table1_xpln_timestamps[row_string] = totalTime
                else:
                    table1_xpln_timestamps[row_string] += totalTime
                if row_string not in xpln_runtimes:
                    xpln_runtimes[row_string] = [totalTime]
                else:
                    xpln_runtimes[row_string].append(totalTime)
            # top option
            elif (isinstance(row, str) and row == "top"):
                row_string = row
                top, _ = query.betters(data, len(best.rows))
                top = DATA(data, top)
                data_for_stats = top
            # all option
            else:
                row_string = row
                data_for_stats = data
            current_stats = query.stats(data_for_stats, mean, includeN = False)
            if not table1_current_row_stats:
                # Initialize the dictionary with the first set of stats
                table1_current_row_stats = {key: value for key, value in current_stats.items()}
            else:
                # Update the dictionary with the cumulative sum for each key
                for key, value in current_stats.items():
                    table1_current_row_stats[key] += value
            obj_values.append(list(current_stats.values()))
            if i == 20:
                for key in table1_current_row_stats.keys():
                    table1_current_row_stats[key] /= i
                table1_current_file_dict[row if isinstance(row, str) else row.__name__] = table1_current_row_stats
                clusters[row_string] = obj_values
                table1_string += f"{row_string:>30}{''.join(f'{round(v, 2):>20}' for v in table1_current_row_stats.values())}\n"

    for key in table1_xpln_timestamps.keys():
        table1_xpln_timestamps[key] /= 20
    print(table1_string)
    sk_string = ""
    for index, col in enumerate(data.cols.y):
        sk_string += f"Scott-Knott for: {col.col.txt}\n"
        rxs = [RX([list[index] for list in clusters[key]], key) for key in clusters.keys()]
        for rx in tiles(scottKnot(rxs)):
            spaces_str = " " * (max_row_header_len - len(rx['name']))
            sk_string += f"{rx['rank']} {rx['name']} {spaces_str} {rx['show']}\n"
        sk_string += "\n"
    print(sk_string)
    with open(full_file_path, "a", encoding="utf-8") as file:
        file.write("Table 1:\n" + table1_string + "\n")
        file.write(sk_string)
    return table1_current_file_dict, table1_xpln_timestamps, xpln_runtimes

def table2(table1_dict, full_file_path):
    col_headers = []
    first_key = next(iter(table1_dict))
    for key in table1_dict[first_key].keys():
        col_headers.append(key)
    table2_string = f"{'':>30}{''.join(f'{h:>20}' for h in col_headers)}\n"
    comparisons = {"all": ["all", "sway1", "sway2"],
                   "sway1": ["sway2", "xpln1", "top"],
                   "sway2": ["xpln2"]
                   }

    for key, listOfComparison in comparisons.items():
        # "all": ["all", "sway1"]"         key is "all"     listOfComparison = ["all", "sway1"]
        for comparisonKey in listOfComparison:
            # comparisonKey is gonna iterate over ["all", "sway1"]
            isEqualTo = []
            for col in col_headers:
                if table1_dict[key][col] == table1_dict[comparisonKey][col]:
                    isEqualTo.append("=")
                else:
                    isEqualTo.append("≠")
            row_string = key + " to " + comparisonKey
            table2_string += f"{row_string:>30}{''.join(f'{value:>20}' for value in isEqualTo)}\n"
    print(table2_string)
    with open(full_file_path, "a", encoding="utf-8") as file:
        file.write("Table 2:\n" + table2_string + "\n")

def xplnRuntimes(xpln_timestamps, full_file_path, xpln_runtimes):
    xplnRuntimes_string = ""
    for key in xpln_timestamps:
        row_string = key + ":"
        time_string = str(round(xpln_timestamps[key] * 1000, 2)) + "ms"
        xplnRuntimes_string += f"{row_string:>10}{''.join(f'{time_string:>20}')}\n"

    print(xplnRuntimes_string)
    sk_string = f"Scott-Knott for: xpln Runtimes\n"
    rxs = [RX([list for list in xpln_runtimes[key]], key) for key in xpln_runtimes.keys()]
    for rx in tiles(scottKnot(rxs)):
        sk_string += f"{rx['rank']} {rx['name']} {rx['show']}\n"
    sk_string += "\n"
    print(sk_string)
    with open(full_file_path, "a", encoding="utf-8") as file:
        file.write("Xpln Runtimes:\n" + xplnRuntimes_string + "\n")
        file.write(sk_string)

def sampleTaxes(table1_dict, full_file_path):
    col_headers = []
    first_key = next(iter(table1_dict))
    for key in table1_dict[first_key].keys():
        col_headers.append(key)
    sampleTaxes_string = f"{'':>30}{''.join(f'{h:>20}' for h in col_headers)}\n"
    all_values = list(table1_dict["all"].values())
    taxList = ["sway1", "sway2"]
    for tax in taxList:
        row_string = f"Sampling Tax (all - {tax})"
        sampleTaxes_string += f"{row_string:>30}{''.join(f'{round(all_values[index] - value, 1):>20}' for index, value in enumerate(list(table1_dict[tax].values())))}\n"

    print(sampleTaxes_string)
    with open(full_file_path, "a", encoding="utf-8") as file:
        file.write("Sampling Taxes:\n" + sampleTaxes_string + "\n")

def xplnTaxes(table1_dict, full_file_path):
    col_headers = []
    first_key = next(iter(table1_dict))
    for key in table1_dict[first_key].keys():
        col_headers.append(key)
    xplnTaxes_string = f"{'':>30}{''.join(f'{h:>20}' for h in col_headers)}\n"
    taxList = [["xpln1", "sway1"], ["xpln2", "sway2"]]
    for tax in taxList:
        xpln = tax[0]
        sway = tax[1]
        sway_values = list(table1_dict[sway].values())
        row_string = f"{xpln} Tax ({sway} - {xpln})"
        xplnTaxes_string += f"{row_string:>30}{''.join(f'{round(sway_values[index] - value, 1):>20}' for index, value in enumerate(list(table1_dict[xpln].values())))}\n"

    xplnTaxes_string += "\n"
    print(xplnTaxes_string)
    with open(full_file_path, "a", encoding="utf-8") as file:
        file.write("Explanation Taxes:\n" + xplnTaxes_string + "\n")

def scottKnot(rxs):
    global args
    def merges(i, j):
        out = RX([], rxs[i]["name"])
        for k in range(i, j + 1):
            out = merge(out, rxs[j])
        return out
    def same(lo, cut, hi):
        l = merges(lo, cut)
        r = merges(cut + 1, hi)
        return cliffsDelta(l["has"], r["has"]) and bootstrap(l["has"], r["has"])
    def recurse(lo, hi, rank):
        cut = None
        b4 = merges(lo, hi)
        best = 0
        for j in range(lo, hi + 1):
            if j < hi:
                l = merges(lo, j)
                r = merges(j + 1, hi)
                now = (l["n"] * (mid(l) - mid(b4)) ** 2 + r["n"] * (mid(r) - mid(b4)) ** 2) / (l["n"] + r["n"])
                if now > best:
                    if abs(mid(l) - mid(r)) > cohen:
                        cut, best = j, now
        if cut and not same(lo, cut, hi):
            rank = recurse(lo, cut, rank) + 1
            rank  = recurse(cut + 1, hi, rank)
        else:
            for i in range(lo, hi + 1):
                rxs[i]["rank"] = rank
        return rank
    rxs.sort(key=lambda x: mid(x))
    cohen = div(merges(0, len(rxs) - 1)) * args.cohen
    recurse(0, len(rxs) - 1, 1)
    return rxs

def merge(rx1, rx2):
    rx3 = RX([], rx1["name"])
    for t in (rx1["has"], rx2["has"]):
        for x in t:
            rx3["has"].append(x)
    rx3["has"].sort()
    rx3["n"] = len(rx3["has"])
    return rx3

def RX(t, s = None):
    t.sort()
    return {"name": s or "", "rank": 0, "n": len(t), "show": "", "has": t}

def mid(t):
    t = t["has"] if "has" in t else t
    n = len(t) // 2
    return len(t) % 2 == 0 and (t[n] + t[n + 1]) / 2 or t[n + 1]

def div(t):
    t = t["has"] if "has" in t else t
    return (t[len(t) * 9 // 10] - t[len(t) * 1 // 10]) / 2.56

def samples(t, n = None):
    u = []
    for i in range(n or len(t)):
        u.append(random.choice(t))
    return u

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
        Whether the difference between the 2 lists of rows is great enough to be non-trivial
    """
    global args
    n, gt, lt = 0, 0, 0
    if len(ns1) > 128: ns1 = samples(ns1, 128)
    if len(ns2) > 128: ns2 = samples(ns2, 128)
    for x in ns1:
        for y in ns2:
            n += 1
            if x > y: gt += 1
            if x < y: lt += 1
    return abs(lt - gt) / n <= args.cliff

def add(i, x):
    i.n += 1
    d = x - i.mu
    i.mu += d / i.n
    i.m2 += d * (x - i.mu)
    i.sd = 0 if i.n < 2 else math.sqrt(i.m2 / (i.n - 1))

def delta(i, other):
    global smallPositive
    e, y, z = smallPositive, i, other
    return abs(y.mu - z.mu) / (math.sqrt(e + y.sd ** 2 / e + y.n + z.sd ** 2 / (z.n + e)))

def bootstrap(y0, z0):
    global args
    x, y, z, yhat, zhat = NUM(), NUM(), NUM(), [], []
    for y1 in y0:
        add(x, y1)
        add(y, y1)
    for z1 in z0:
        add(x, z1)
        add(z, z1)
    xmu, ymu, zmu = x.mu, y.mu, z.mu
    for y1 in y0: yhat.append(y1 - ymu + xmu)
    for z1 in z0: zhat.append(z1 - zmu + xmu)
    tobs = delta(y, z)
    n = 0
    for i in range(args.bootstrap):
        if (delta(NUM(samples(yhat)), NUM(samples(zhat))) > tobs):
            n += 1
    return n / args.bootstrap > args.conf

def tiles(rxs):
    global smallPositive, args
    huge, minF, maxF, floor = float("inf"), min, max, math.floor
    lo, hi = huge, -huge
    for rx in rxs:
        lo, hi = minF(lo, rx["has"][0]), maxF(hi, rx["has"][-1])
    for rx in rxs:
        t, u = rx["has"], []
        def of(x, most): return max(1, minF(most, x))
        def at(x):
            return t[of(int(len(t) * x), len(t) - 1)]
        def pos(x): return floor(of(args.width * (x - lo) / (hi - lo + smallPositive) // 1, args.width))
        for _ in range(args.width): u.append(" ")
        a, b, c, d, e= at(.1), at(.3), at(.5), at(.7), at(.9)
        A, B, C, D, E= pos(a), pos(b), pos(c), pos(d), pos(e)
        for i in range(A, B):
            u[i - 1] = "-"
        for i in range(D, E):
            u[i - 1] = "-"
        u[(args.width // 2) - 1] = "|"
        u[C - 1] = "*"
        rx["show"] = "".join(u) + " { %6.2f" % a
        for x in (b, c, d, e):
            rx["show"] += ", %6.2f" % x
        rx["show"] += "  }"
    return rxs
