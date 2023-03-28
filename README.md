# Final Project

![Task](https://img.shields.io/badge/Task-FinalProject-blue.svg)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ConnorS1110/csc591-ase-final-project/test.yml?label=Tests&logo=github)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/github/languages/top/ConnorS1110/csc591-ase-final-project.svg)](https://github.com/ConnorS1110/csc591-ase-final-project)
[![DOI](https://zenodo.org/badge/607756153.svg)](https://zenodo.org/badge/latestdoi/607756153)
![GitHub contributors](https://img.shields.io/github/contributors/ConnorS1110/csc591-ase-final-project?label=Contributors&logo=github)

This program is an argument parser that interprets command-line arguments, and outputs information depending on the flags that are set. A series of pre-made examples are run that run nested functions for each example and determines whether the output of those nested functions is `True` or `False`. Depending on the flags that are set, you can see the output of these test results in the terminal window. Now supports the use of the `-f` or `--file` arguments that take the relative path of a csv file for statistics to be run on the file. After the data has been read in, the data is clustered by recursively splitting the data in half that has been sorted by distance from a random row until clusters are created of a pre-determined size. Statistics are generated for that cluster make predictive statements about the data in the cluster. The data can also be swayed to make a prediction of the best row in the data by recursively clustering the data.

## Instructions

The examples are run from `main.py` in the src directory. To run the code, at the root directory, run the following command in a terminal window:

```
python src/main.py [FLAG] [VALUE]
```

For example, to see the results of all the test cases run:

```
python src/main.py -g all
```

To use the new `-f` command run:

```
python src/main.py -g all -f ../etc/data/auto93.csv
```
`Note this flag is not required and uses this relative path as a default`

## Team Members

- Connor Smith (Unity ID: cpsmith6)
- Ashvin Gaonkar (Unity ID: agaonka2)
- Liwen "Cynthia" Du (Unity ID: ldu2)
