
'''
python hard.py 2000 0.1 0.1 0 &
python hard.py 2000 0.1 0.1 1 &
python hard.py 2000 0.1 0.1 2 &
python hard.py 2000 0.1 0.1 3 &
python hard.py 2000 0.1 0.1 4 &
python hard.py 2000 0.1 0.1 5 &
python hard.py 2000 0.1 0.1 6 &

import numpy as np
for i in range(7):
    modeli = np.load(f'result/gl_res{i}.npz')
    np.mean(modeli['P']!=0)

'''

import os
import sys

niter = int(sys.argv[1])
alpha = float(sys.argv[2])
lmd = float(sys.argv[3])
datai = int(sys.argv[4])

ncore = '4'
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore

import numpy as np
import dataseting
import pandas as pd
from mblr import *
import matplotlib.pyplot as plot

def plotts(x): plot.plot(x); plot.show()

def loadres():
    X = pd.read_csv('data/resx.csv', index_col=0)
    Y = pd.read_csv('data/resy.csv',index_col=0)
    age = pd.read_csv('data/age.csv',index_col=0)['age']
    Xindex = X.columns
    Yindex = Y.columns
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y,age,Xindex,Yindex

X,Y,age,Xindex,Yindex = loadres()

agetier = []
for i in range(7):
    agei = (age >= i * 5 + 20) & (age <= i * 5 + 30)
    print(sum(agei))
    agetier.append(agei)

datasets = dataseting.split_dataset2(X.T,Y.T,agetier)

if __name__ == '__main__':
    i = datai
    X = datasets[i][0]
    Y = datasets[i][1]
    V, P, W, k, lls = hard2(X, Y, niter, alpha=alpha, rho=40.39, lmd=lmd, tol=1e-8, verbose=0)
    np.savez(f'result/gl_res{i}.npz', V=V, P=P, W=W, k=k, lls=lls)