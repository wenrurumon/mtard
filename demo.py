
import os

ncore = '16'
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore

import numpy as np
from mblr import *

demofiles = np.load('data/demofiles.npz')
X = demofiles['X']
Y = demofiles['Y']
hardz(X, Y, 10, alpha=.1, rho=.1, lmd=.1, tol=1e-7, verbose=1, stops=20, code=f'result//demo')