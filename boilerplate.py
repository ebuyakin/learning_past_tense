# initial import and settings (template)

import sys
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from time import time
from random import sample

from nltk.corpus import wordnet as wn
from anytree import Node, RenderTree
from pprint import pprint as pp  # pretty printing


# plotting output settings
pio.renderers.default = 'chrome'
pio.templates.default = 'plotly_dark'

# console printout settings:
np.set_printoptions(precision=3, linewidth=200, suppress=True)
pd.set_option('display.max_columns', None)  # manage the number of columns that are printed into console
pd.set_option('display.precision', 3)  # floating point output precision (number of significant digits)
pd.set_option('display.width', None)  # the setting for the total width of the dataframe as it's printed.
pd.set_option('display.max_rows', 100)  # number of rows that is printed without truncation


# set matplotlib params
mpl.rcParams['figure.dpi'] = 500


# pretty printing function
def ppp(x, n=5):  # print list x by n items per line
    i = 0
    for i in range(int(len(x)/n)):
        print(x[i*n:i*n+n])
    print(x[(i+1)*n:])
    print('total: ', len(x))
