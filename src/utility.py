import argparse
import csv
import json
import math
import os
from data import DATA
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
    with open(sFilename, mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            fun(line)

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
                          "../etc/data/nasa93dem.csv", "../etc/data/pom.csv", "../etc/data/SSM.csv", "../etc/data/SSN.csv", ]
    for file in list_of_file_paths:
        print(f"File: {str(file.split('/')[-1]).split('.')[0]}")
        table1(file)

def table1(filepath):
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, filepath)
    data = DATA(full_path)
    row_headers = ["all", opt.sway1, disc.xpln1, "top"]
    col_headers = []
    for col in data.cols.y:
        col_headers.append(col.col.txt)
    table1_string = f"{'':>20}{''.join(f'{h:>14}' for h in col_headers)}\n"
    for row in row_headers:
        table1_current_row_stats = {}
        for i in range(1, 21):
            data = DATA(full_path)
            # sway1 option
            if (not isinstance(row, str) and row.__name__ == "sway1"):
                row_func = row
                row_string = row.__name__
                best, _, _ = row_func(data)
                data_for_stats = best
            # xpln1 option
            elif (not isinstance(row, str) and row.__name__ == "xpln1"):
                row_func = row
                row_string = row.__name__
                rule = None
                while (rule == None):
                    best, rest, _ = opt.sway1(data)
                    rule, _ = row_func(data, best, rest, False)
                data1 = DATA(data, disc.selects(rule, data.rows))
                data_for_stats = data1
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
            current_stats = query.stats(data_for_stats, includeN = False)
            if not table1_current_row_stats:
                # Initialize the dictionary with the first set of stats
                table1_current_row_stats = {key: value for key, value in current_stats.items()}
            else:
                # Update the dictionary with the cumulative sum for each key
                for key, value in current_stats.items():
                    table1_current_row_stats[key] += value
            if i == 20:
                for key in table1_current_row_stats.keys():
                    table1_current_row_stats[key] /= i
                table1_string += f"{row_string:>20}{''.join(f'{round(v, 2):>14}' for v in table1_current_row_stats.values())}\n"

    print(table1_string)
