
import os
import dataseting
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler

ncore = '8'
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore

import numpy as np
from mblr import *
import matplotlib.pyplot as plot

def plotts(x): plot.plot(x); plot.show()

def loaddata():
    X = pd.read_csv('data/X20706.csv', index_col=0)
    Y = pd.read_csv('data/Y20706.csv',index_col=0)
    X = np.log(X+1e-16)
    Z = pd.read_csv('data/Z20706.csv',index_col=0)
    Xindex = X.columns
    Yindex = Y.columns
    Zindex = Z.columns
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    return X,Y,Z,Xindex,Yindex,Zindex

X,Y,Z,Xindex,Yindex,Zindex = loaddata()
age = Z[:,1]

################################################################
# Modeling
################################################################

#Age Tier

agetier = []
agetier.append(age>0)
y = 2
for i in range(int(30/y+1)):
    agei = (age >= i * y + 20) & (age <= i * y + 30)
    print(i * y + 25)
    agetier.append(agei)

#Dataset

datasets = dataseting.split_XYZ(X,Y,Z,agetier)

Y = datasets[0][1]
covs = []
for i in range(len(datasets)):
    print(i)
    covi = []
    for j in range(100):
        Yi = Y[np.random.choice(Y.shape[0], np.sum(agetier[i]), replace=False),]
        scaler =  StandardScaler()
        Yi = scaler.fit_transform(Yi)
        print(Yi.shape,np.quantile(abs(np.cov(Yi)),0.98))
        covi.append(np.quantile(abs(np.cov(Yi)),0.98))
    covs.append(covi)

covs2 = []
for i in covs: covs2.append(np.mean(i))

for i in range(len(datasets)):
    X = np.exp(datasets[i][0])
    Y = datasets[i][1]
    Z = datasets[i][2]
    Z[:,1] = 1
    model = sklearn.linear_model.LinearRegression()
    model.fit(Z, X)
    X2 = np.log(X)
    X2 = X - model.predict(Z)
    model.fit(Z,Y)
    # Y2 = Y
    Y2 = Y - model.predict(Z)
    scaler =  StandardScaler()
    X2 = scaler.fit_transform(X2)
    Y2 = scaler.fit_transform(Y2)
    hardz2(X2.T, Y2.T, 10000, alpha=.1, rho=covs2[i], lmd=.1, tol=1e-6, verbose=1, stops=10000, code=f'result2//model{i}')
    # if i==0:
    #     print(i)
    #     hardz2(X2.T, Y2.T, 1450, alpha=.1, rho=covs2[i], lmd=.1, tol=0, verbose=1, stops=10000, code=f'result2//model{i}')
    # else:
    #     print(i)
    #     hardz2(X2.T, Y2.T, i*25+1200, alpha=.1, rho=covs2[i], lmd=.1, tol=0, verbose=1, stops=10000, code=f'result2//model{i}')
