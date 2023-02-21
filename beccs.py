## ROBUST DECISION MAKING (RDM) MODEL FOR BECCS
# Author: Oscar Stenström
# Date: 2023-02-16


## MODEL BEGINS HERE
import math
import numpy as np
import numpy_financial as npf
from scipy.optimize import brentq as root
from rhodium import (
    Model,
    Parameter,
    Response,
    RealLever,
    UniformUncertainty,
    sample_lhs,
    update,
    evaluate,
    scatter2d,
    Cart,
    pairs,
    sa,
)
import pandas as pd
import csv
import openpyxl
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
