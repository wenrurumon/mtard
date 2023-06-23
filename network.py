
import os
import dataseting
import pandas as pd
import sklearn.linear_model

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
    X = pd.read_csv('data/data0211/Xrna.csv', index_col=0)
    Y = pd.read_csv('data/data0211/Ystruct.csv',index_col=0)
    Z = pd.read_csv('data/data0211/Z.csv',index_col=0)
    Xindex = X.columns
    Yindex = Y.columns
    Zindex = Z.columns
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    return X,Y,Z,Xindex,Yindex,Zindex

def loaddata2():
    X = pd.read_csv('data/data0211/Xrna.csv', index_col=0)
    Y = pd.read_csv('data/data0211/Ystruct.csv',index_col=0)
    X = pd.concat((X,Y),axis=1)
    Y = pd.read_csv('data/data0211/Yfunc.csv',index_col=0)
    Z = pd.read_csv('data/data0211/Z.csv',index_col=0)
    Xindex = X.columns
    Yindex = Y.columns
    Zindex = Z.columns
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    return X,Y,Z,Xindex,Yindex,Zindex

def loaddata3():
    X = pd.read_csv('data/data0211/Xrna.csv', index_col=0)
    Y = pd.read_csv('data/data0211/Y.csv',index_col=0)
    Z = pd.read_csv('data/data0211/Z.csv',index_col=0)
    Xindex = X.columns
    Yindex = Y.columns
    Zindex = Z.columns
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    return X,Y,Z,Xindex,Yindex,Zindex

def loaddata4():
    X = pd.read_csv('data/data0211/Xage2.csv', index_col=0)
    Y = pd.read_csv('data/data0211/Y.csv',index_col=0)
    Z = pd.read_csv('data/data0211/Z.csv',index_col=0)
    Xindex = X.columns
    Yindex = Y.columns
    Zindex = Z.columns
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    return X,Y,Z,Xindex,Yindex,Zindex

################################################################
# Modeling
################################################################

rhos = [0.187977042012282,0.1997777983383,0.197666494019576,0.200735321801914,0.208483841638718,0.214200406889276,0.214915171002125,0.220681296898416,0.223260293941547,0.228084124167506,0.229580117075648,0.235737808555536,0.236205227759274,0.243630056359296,0.251734526789372,0.25357232276954,0.259805005164879]

#loaddata

X,Y,Z,Xindex,Yindex,Zindex = loaddata()
age = Z[:,0]

agetier = []
agetier.append(age>0)
y = 2
for i in range(int(30/y+1)):
    agei = (age >= i * y + 20) & (age <= i * y + 30)
    print(i * y + 25)
    agetier.append(agei)

datasets = dataseting.split_XYZ2(X,Y,Z,agetier)

for i in range(len(datasets)):
    X = datasets[i][0]
    Y = datasets[i][1]
    Z = datasets[i][2]
    model = sklearn.linear_model.LinearRegression()
    model.fit(Z.T, X.T)
    X2 = sklearn.preprocessing.scale(X - model.predict(Z.T).T, axis=1)
    model.fit(Z.T,Y.T)
    Y2 = sklearn.preprocessing.scale(Y - model.predict(Z.T).T, axis=1)
    hardz(X2, Y, 5000, alpha=.1, rho=rhos[i], lmd=.1, tol=1e-6, verbose=1, stops=20, code=f'result/result0620/dxdsr09s{i}')

#loaddata2

X,Y,Z,Xindex,Yindex,Zindex = loaddata2()
age = Z[:,0]

agetier = []
agetier.append(age>0)
y = 2
for i in range(int(30/y+1)):
    agei = (age >= i * y + 20) & (age <= i * y + 30)
    print(i * y + 25)
    agetier.append(agei)

datasets = dataseting.split_XYZ2(X,Y,Z,agetier)

for i in range(len(datasets)):
    X = datasets[i][0]
    Y = datasets[i][1]
    Z = datasets[i][2]
    model = sklearn.linear_model.LinearRegression()
    model.fit(Z.T, X.T)
    X2 = sklearn.preprocessing.scale(X - model.predict(Z.T).T, axis=1)
    model.fit(Z.T,Y.T)
    Y2 = sklearn.preprocessing.scale(Y - model.predict(Z.T).T, axis=1)
    hardz(X2, Y, 5000, alpha=.1, rho=rhos[i], lmd=.1, tol=1e-6, verbose=1, stops=20, code=f'result/result0620/dxsdfr09s{i}')

#loaddata3

X,Y,Z,Xindex,Yindex,Zindex = loaddata3()
age = Z[:,0]

agetier = []
agetier.append(age>0)
y = 2
for i in range(int(30/y+1)):
    agei = (age >= i * y + 20) & (age <= i * y + 30)
    print(i * y + 25)
    agetier.append(agei)

datasets = dataseting.split_XYZ2(X,Y,Z,agetier)

for i in range(len(datasets)):
    X = datasets[i][0]
    Y = datasets[i][1]
    Z = datasets[i][2]
    model = sklearn.linear_model.LinearRegression()
    model.fit(Z.T, X.T)
    X2 = sklearn.preprocessing.scale(X - model.predict(Z.T).T, axis=1)
    model.fit(Z.T,Y.T)
    Y2 = sklearn.preprocessing.scale(Y - model.predict(Z.T).T, axis=1)
    hardz(X2, Y, 5000, alpha=.1, rho=rhos[i], lmd=.1, tol=1e-6, verbose=1, stops=20, code=f'result/result0620/dxdsfr09s{i}')

#loaddata3

X,Y,Z,Xindex,Yindex,Zindex = loaddata3()
age = Z[:,0]

agetier = []
agetier.append(age>0)
y = 2
for i in range(int(30/y+1)):
    agei = (age >= i * y + 20) & (age <= i * y + 30)
    print(i * y + 25)
    agetier.append(agei)

datasets = dataseting.split_XYZ2(X,Y,Z,agetier)

for i in range(len(datasets)):
    X = datasets[i][0]
    Y = datasets[i][1]
    Z = datasets[i][2]
    model = sklearn.linear_model.LinearRegression()
    model.fit(Z.T, X.T)
    X2 = sklearn.preprocessing.scale(X - model.predict(Z.T).T, axis=1)
    model.fit(Z.T,Y.T)
    Y2 = sklearn.preprocessing.scale(Y - model.predict(Z.T).T, axis=1)
    hardz(X2, Y, 5000, alpha=.1, rho=1, lmd=.1, tol=1e-6, verbose=1, stops=20, code=f'result/result0620/dxdsfr00s{i}')

#loaddata4

X,Y,Z,Xindex,Yindex,Zindex = loaddata4()
age = Z[:,0]

agetier = []
agetier.append(age>0)
y = 2
for i in range(int(30/y+1)):
    agei = (age >= i * y + 20) & (age <= i * y + 30)
    print(i * y + 25)
    agetier.append(agei)

datasets = dataseting.split_XYZ2(X,Y,Z,agetier)

for i in range(len(datasets)):
    X = datasets[i][0]
    Y = datasets[i][1]
    Z = datasets[i][2]
    model = sklearn.linear_model.LinearRegression()
    model.fit(Z.T, X.T)
    X2 = sklearn.preprocessing.scale(X - model.predict(Z.T).T, axis=1)
    model.fit(Z.T,Y.T)
    Y2 = sklearn.preprocessing.scale(Y - model.predict(Z.T).T, axis=1)
    hardz(X2, Y, 5000, alpha=.1, rho=1, lmd=.1, tol=1e-6, verbose=1, stops=20, code=f'result/result0620/dx2dsfr00s{i}')

