
import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
import numpy as np
import random
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
eps = np.finfo(float).eps

def load_hexin0707():
    X = pd.read_csv('data/hexin0708/X.csv',index_col=0)
    Y = pd.read_csv('data/hexin0708/Y.csv',index_col=0)
    A = pd.read_csv('data/hexin0708/A.csv',index_col=0)
    Z = pd.read_csv('data/hexin0708/E.csv',index_col=0)
    xindex = X.index
    yindex = Y.index
    aindex = A.index
    zindex = Z.index
    X = np.asarray(X)
    Y = np.asarray(Y)
    A = np.asarray(A)
    Z = np.asarray(Z)
    sel = np.isnan(A.sum(axis=0)) == False
    X = X[:,sel]
    Y = Y[:,sel]
    A = A[:,sel]
    Z = Z[:,sel]
    for i in range(X.shape[0]):
        X[i,np.isnan(X[i,:])] = np.median(X[i,np.isnan(X[i,:])==False])
    age = A[0,:]
    female = Z[0,:]==1
    return X,Y,A,Z,xindex,yindex,aindex,zindex,age,female

def split_XYZ(X,Y,Z,labels):
    datasets = []
    for i in labels:
        Xi = X[i,:]
        Yi = Y[i,:]
        Zi = Z[i,:]
        scaler = StandardScaler()
        Xi = scaler.fit_transform(Xi)
        Yi = scaler.fit_transform(Yi)
        Zi = scaler.fit_transform(Zi)
        datasets.append((Xi,Yi,Zi))
    return datasets

def split_XYZ2(X,Y,Z,labels):
    datasets = []
    for i in labels:
        Xi = epsit(X[i,:])
        Yi = epsit(Y[i,:])
        Zi = Z[i,:]
        Xmean = Xi.mean(axis=0, keepdims=True)
        Xsd = Xi.std(axis=0, keepdims=True)
        # Xsd = 0 if Xsd == 0 else 1 / Xsd
        Xsd = np.where(Xsd == 0, 0, 1 / Xsd)
        Xi = (Xi-Xmean)*Xsd
        Ymean = Yi.mean(axis=0, keepdims=True)
        Ysd = Yi.std(axis=0, keepdims=True)
        # Ysd = 0 if Ysd == 0 else 1/Ysd
        Ysd = np.where(Ysd == 0, 0, 1 / Ysd)
        Yi = (Yi-Ymean)*Ysd
        Zmean = Zi.mean(axis=0, keepdims=True)
        Zsd = Zi.std(axis=0, keepdims=True)
        # Zsd = 0 if Zsd == 0 else 1/Zsd
        Zsd = np.asarray([1/np.ravel(Zsd)[0],1,1])
        Zi = (Zi-Zmean)*Zsd
        datasets.append((Xi.T,Yi.T,Zi.T))
    return datasets

def split_dataset(X,Y,seed,nfold,scaleX=True,scaleY=False):
    # datasets = split_dataset(X,Y,0,5)
    random.seed(seed)
    N = X.shape[1]
    sample = np.ravel(random.sample(range(N),N)) / N
    datasets = []
    for i in range(nfold):
        tr = (sample<(i*1/nfold)) | (sample>=((i+1)*1/nfold))
        te = (sample>=(i*1/nfold)) & (sample<((i+1)*1/nfold))
        if scaleX:
            Xmean = X[:,tr].mean(axis=1,keepdims=True)
            Xsd = X[:,tr].std(axis=1,keepdims=True)+eps
            X = (X-Xmean)/Xsd
        else:
            Xmean = 1
            Xsd = 1
        if scaleY:
            Ymean = Y[:,tr].mean(axis=1,keepdims=True)
            Ysd = Y[:,tr].std(axis=1,keepdims=True)+eps
            Y = (Y-Ymean)/Ysd
        else:
            Ymean = 1
            Ysd = 1
        Xtr = X[:,tr]
        Ytr = Y[:,tr]
        Xte = X[:,te]
        Yte = Y[:,te]
        # print(i,Xtr.shape,Ytr.shape,Xte.shape,Yte.shape)
        datasets.append((Xtr,Ytr,Xte,Yte,Xmean,Ymean,Xsd,Ysd))
    return datasets

def improve(X,size,seed):
    if(X.shape[1]>=size):
        X2 = sklearn.preprocessing.scale(X, axis=1)
    else:
        np.random.seed(seed)
        N = X.shape[1]
        sel = np.random.choice(range(N),size-N)
        X2 = X[:,sel]
        E2 = np.random.normal(0,0.1,X2.shape)
        X2 = X2 + E2
        X2 = np.concatenate((X,X2),axis=1)
        X2 = sklearn.preprocessing.scale(X2,axis=1)
    return(X2)

def epsit(X,seed=1,sd=0.0001):
    np.random.seed(seed)
    E = np.random.normal(0,sd,X.shape)
    X = X + E
    X2 = sklearn.preprocessing.scale(X,axis=1)
    return(X2)

def split_dataset2(X,Y,labels):
    datasets = []
    for i in labels:
        sel = i
        Xi = X[:, sel]
        Yi = Y[:, sel]
        Xmean = Xi.mean(axis=1,keepdims=True)
        Xsd = Xi.std(axis=1,keepdims=True)+eps
        Ymean = Yi.mean(axis=1,keepdims=True)
        Ysd = Yi.std(axis=1,keepdims=True)+eps
        Xi = (Xi-Xmean)/Xsd
        Yi = (Yi-Ymean)/Ysd
        Xi[np.isnan(Xi)] = 1
        Yi[np.isnan(Yi)] = 1
        # print(Xi.shape,Yi.shape)
        datasets.append((Xi,Yi))
    return datasets


