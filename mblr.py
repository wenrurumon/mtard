import os
import sys
import random
# import rpy2.robjects.numpy2ri
# from rpy2.robjects.packages import importr
# rpy2.robjects.numpy2ri.activate()
# glassoFast = rpy2.robjects.packages.importr('glassoFast').glassoFast
import numpy as np
import scipy as sp
from numpy import log as log
from numpy import eye as eye
from numpy import diag as diag
from numpy import sqrt as sqrt
from numpy import ones as ones
from numpy import zeros as zeros
from numpy import outer as outer
from numpy import trace as trace
from numpy import cumprod as cumprod
from numpy import identity as identity
from numpy import concatenate as concat
from numpy.linalg import det as det
from numpy.linalg import inv as inv
from numpy.linalg import svd as svd
from numpy.linalg import norm as norm
from numpy.linalg import solve as solve
from numpy.linalg import eigvals as eigvals
from numpy.linalg import slogdet as slogdet
from numpy.linalg import cholesky as cholesky
from numpy.linalg import matrix_rank as matrix_rank
from numpy.random import normal as normal
from numpy.random import uniform as uniform
from numpy.random import multivariate_normal as multivariate_normal
from scipy.stats import matrix_normal
from scipy.linalg import eigh
from scipy.linalg import eigvalsh
from scipy.linalg import pinvh
import datetime
import fglasso
import time

import warnings
warnings.simplefilter("error")
eps = np.finfo(float).eps

float_formatter = "{:.2e}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

def getidx(x):
    return [i for i in range(len(x)) if x[i]]

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def is_pd(V):
    return np.all(eigvalsh(V) > 0)


def is_psd(V):
    return np.all(eigvalsh(V) >= 0)

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = svd(B)
    H = np.dot(V.T, np.dot(diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    return A3

def matreg(V0, tol=1e-9):
    V = V0.copy()
    r = matrix_rank(V)
    while r < V.shape[0]:
        # print('𐄂 V low rank:', r, '/', V.shape[0])
        V = shrinkmat(V, tol)
        # from sklearn.covariance import shrunk_covariance
        # V = shrunk_covariance(V, 0.05)
        r = matrix_rank(V)
    # print('✓ V rank:', r, '/', V.shape[0], ', PD:', isPD(V))
    return V

def matthres(V0,vk):
    V = V0.copy()
    V0 = V0 * (abs(V0)>vk)
    V0[range(V.shape[1]),range(V.shape[1])] = diag(V)
    return V0

def glasso(V,rho,maxit):
    if rho==0:
        P = 0
        return V,P
    else:
        return fglasso.fglasso(V,rho,maxit)

def shrinkmat(V0, tol):
    # method1
    # L,sval,R = svd(V)
    # sval = sval*(sval>1e-6)+(sval+1e-6)*(sval<1e-6)
    # return L @ diag(sval) @ R
    # method2
    # return V + 1e-6 * identity(M)
    # method3
    return (1 - tol) * V0 + tol * trace(V0) / V0.shape[0] * identity(
        V0.shape[0])

def llh_hard(X, S, V, k):
    # simplify using matrix inverse lemma and determinant lemma
    # XXK is XX + diag(1 / k)
    # S is YY - YXinvXXK @ YX.T
    N = X.shape[1]
    M = V.shape[0]
    return - N / 2 * cumprod(slogdet(V))[-1] \
           - M / 2 * (cumprod(slogdet(identity(N) + X.T * k @ X))[-1]) \
           - 0.5 * trace(solve(V, S))

def hard(X,Y,niter,alpha=0.1,rho=0.1,lmd=0.1,tol=1e-6,verbose=0,stops=100):
    now = datetime.datetime.now()
    D,N = X.shape
    M = Y.shape[0]
    P = 0
    XX = X @ X.T
    YX = Y @ X.T
    YY = Y @ Y.T
    if rho < 1:
        rho = np.quantile(abs(YY / N), 1 - rho)
    else:
        rho = rho-1
    k = ones(D) * alpha
    V = matreg((YY - YX @ solve(XX + diag(1 / k), YX.T)) / N)
    lls = []
    lastll = -np.inf
    model = []
    stop = 0
    converge = 0
    for i in range(niter):
        # accelerate using matrix inverse lemma,
        # pay attention to numerical accuracy
        XK = X.T * k
        invXXK = diag(k) - XK.T @ inv(identity(N) + X.T * k @ X) @ XK
        YXinvXXK = YX @ invXXK
        S = YY - YXinvXXK @ YX.T
        S = (S+S.T)/2
        V = matreg(1 / N * S)
        # loglikelihood
        if i==0:
            ll = -np.inf
        else:
            ll = llh_hard(X, S, V, k) - rho * gloss
        lls.append(ll)
        if verbose > 0:
            print(i, '@l =', ll, '@',datetime.datetime.now())
        if ll<lastll:
            stop += 1
            print(stop)
            if stop > stops:
                converge = 2
                break
        else:
            stop = 0
        if abs(ll - lastll) < tol * abs(lastll):
            lastll = ll
            converge = 1
            break
        else:
            lastll = ll
        # regularize V
        if rho != 0:
            if i==0:
                V, P = glasso(matreg(V), rho, 1)
            else:
                V,P = glasso(matreg(V),rho,1)
            print(np.mean(P!=0))
        gloss = np.sum(abs(P))
        # sparsify invV and return invV
        # update K using posterior of W
        k = diag(invXXK) + 1 / M * (YXinvXXK.T * solve(V, YXinvXXK).T).sum(-1)
        model.append(k)
        if lmd > 0:
            ksqrt = 1 + 8 * lmd * k
            k = (np.sqrt(ksqrt*(ksqrt>0))-1)/(4*lmd)
    invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
    W = Y @ X.T @ invXXK
    return V, P, W, k, lls

def sqrtm(A):
    e, v = eigh(A)
    e[e<0] = 0
    return v @ diag(sqrt(e)) @ v.T

def fa(V,iDim):
    D = V.shape[0]
    r = matrix_rank(V)
    assert(iDim <= r)
    V = matreg(V)
    assert(isPD(V))
    sqrtV = sqrtm(V)
    psi = 1e-12 * diag(V)
    lastpsi = np.zeros(psi.shape)
    while(norm(psi-lastpsi)/len(lastpsi) > 1e-14):
        w,v = eigh(V - diag(psi), subset_by_index=[D-iDim,D-1])
        F = sqrtV @ np.real(v)
        lastpsi = psi
        psi = diag(V) - diag(F @ F.T)
    return F,psi

def myinv(V):
    D = V.shape[0]
    iDim = matrix_rank(V)
    V = matreg(V)
    assert (isPD(V))
    sqrtV = sqrtm(V)
    psi = 1e-12 * diag(V)
    lastpsi = np.zeros(psi.shape)
    while (norm(psi - lastpsi) / len(lastpsi) > 1e-14):
        w, v = eigh(V - diag(psi), subset_by_index=[D - iDim, D - 1])
        F = sqrtV @ np.real(v)
        lastpsi = psi
        psi = diag(V) - diag(F @ F.T)
    #V = psi + F @ F.T
    A = np.diag(1/psi)
    C = np.diag(np.ones(iDim))
    U = F
    V = F.T
    return A - A @ U @ inv(C + V@A@U) @ V @ A

def validbic(f,rho0,loc):
    # loc = '/Users/wenrurumon/Documents/postdoc/mtard/result/m0720'
    model = np.load(f'{loc}/{f}',allow_pickle=True)
    X = model['X']
    Y = model['Y']
    V = model['V']
    k = model['k']
    D, N = X.shape
    M = Y.shape[0]
    YX = Y @ X.T
    YY = Y @ Y.T
    rho = np.quantile(abs(YY / N), rho0)
    XK = X.T * k
    invXXK = diag(k) - XK.T @ inv(identity(N) + X.T * k @ X) @ XK
    YXinvXXK = YX @ invXXK
    # V, P = glassoFast.glassoFast(V, rho)[0:2]
    V, P = fglasso.fglasso(V, rho, 10000)
    k = diag(invXXK) + 1 / M * (YXinvXXK.T * solve(V, YXinvXXK).T).sum(-1)
    invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
    W = Y @ X.T @ invXXK
    YXinvXXK = YX @ invXXK
    S = Y @ Y.T - YXinvXXK @ (Y @ X.T).T
    S = (S + S.T) / 2
    ll = llh_hard(X, S, V, k)
    bic = -2 * ll + log(X.shape[1])*np.sum(P!=0)
    return [rho0,rho,np.mean(P!=0),bic]

def hardz(X,Y,niter,alpha=0.1,rho=0.1,lmd=0.1,tol=1e-6,verbose=0,stops=100,code='temp'):
    #Modeling
    now = datetime.datetime.now()
    D,N = X.shape
    M = Y.shape[0]
    P = 0
    XX = X @ X.T
    YX = Y @ X.T
    YY = Y @ Y.T
    if rho < 1:
        rho = np.quantile(abs(YY / N), 1 - rho)
    else:
        rho = rho-1
    k = ones(D) * alpha
    V = matreg((YY - YX @ solve(XX + diag(1 / k), YX.T)) / N)
    lls = []
    lastll = -np.inf
    stop = 0
    for i in range(niter):
        # accelerate using matrix inverse lemma,
        # pay attention to numerical accuracy
        XK = X.T * k
        print(datetime.datetime.now(),i,'invxxk start')
        invXXK = diag(k) - XK.T @ inv(identity(N) + X.T * k @ X) @ XK
        print(datetime.datetime.now(),i,'invxxk end')
        YXinvXXK = YX @ invXXK
        S = YY - YXinvXXK @ YX.T
        S = (S+S.T)/2
        V = matreg(1 / N * S)
        # loglikelihood
        if i==0:
            ll = -np.inf
        else:
            ll = llh_hard(X, S, V, k) - rho * gloss
        lls.append(ll)
        if verbose > 0:
            print(i, '@l =', ll, '@',datetime.datetime.now(), stop)
        if ll<lastll:
            invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
            W = Y @ X.T @ invXXK
            if stop == 0:
                np.savez(f'{code}_i{i}.npz', P=P, k=k, W=W, lls=lls)
            stop += 1
            if stop > stops:
                break
        else:
            stop = 0
        if abs(ll - lastll) < tol * abs(lastll):
            lastll = ll
            if ll < lastll:
                invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
                W = Y @ X.T @ invXXK
                np.savez(f'{code}_i{i}.npz', P=P, k=k, W=W, lls=lls)
            break
        else:
            lastll = ll
        # regularize V
        if rho != 0:
            print(datetime.datetime.now(),i,'glasso start')
            V, P = glasso(matreg(V), rho, 1)
            print(datetime.datetime.now(),i,'glasso end')
        gloss = np.sum(abs(P))
        # sparsify invV and return invV
        # update K using posterior of W
        k = diag(invXXK) + 1 / M * (YXinvXXK.T * solve(V, YXinvXXK).T).sum(-1)
        if lmd > 0:
            ksqrt = 1 + 8 * lmd * k
            k = (np.sqrt(ksqrt*(ksqrt>0))-1)/(4*lmd)
    invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
    W = Y @ X.T @ invXXK
    np.savez(f'{code}_i{i}.npz',V=V,P=P,k=k,W=W,lls=lls)

def hardz2(X,Y,niter,alpha=0.1,rho=0.1,lmd=0.1,tol=1e-6,verbose=0,stops=100,code='temp'):
    #Modeling
    now = datetime.datetime.now()
    D,N = X.shape
    M = Y.shape[0]
    P = 0
    XX = X @ X.T
    YX = Y @ X.T
    YY = Y @ Y.T
    if rho < 1:
        rho = np.quantile(abs(YY / N), 1 - rho)
    else:
        rho = rho-1
    k = ones(D) * alpha
    V = matreg((YY - YX @ solve(XX + diag(1 / k), YX.T)) / N)
    # V = (YY - YX @ solve(XX + diag(1 / k), YX.T)) / N
    lls = []
    lastll = -np.inf
    stop = 0
    for i in range(niter):
        # accelerate using matrix inverse lemma,
        # pay attention to numerical accuracy
        XK = X.T * k
        invXXK = diag(k) - XK.T @ inv(identity(N) + X.T * k @ X) @ XK
        YXinvXXK = YX @ invXXK
        S = YY - YXinvXXK @ YX.T
        S = (S+S.T)/2
        V = matreg(1 / N * S)
        # V = 1 / N * S
        # loglikelihood
        if i==0:
            ll = -np.inf
        else:
            start = time.time()
            ll = llh_hard(X, S, V, k) - rho * gloss
            # print('lld time:', time.time() - start)
        lls.append(ll)
        if verbose > 0:
            print(i, '@l =', ll, '@',datetime.datetime.now(), stop)
        if ll<lastll:
            invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
            W = Y @ X.T @ invXXK
            if stop == 0:
                np.savez(f'{code}_i{i}.npz', P=P, k=k, W=W, lls=lls)
            stop += 1
            if stop > stops:
                break
        else:
            stop = 0
        if abs(ll - lastll) < tol * abs(lastll):
            lastll = ll
            if ll < lastll:
                invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
                W = Y @ X.T @ invXXK
                np.savez(f'{code}_i{i}.npz', P=P, k=k, W=W, lls=lls)
            break
        else:
            lastll = ll
        # regularize V
        if rho != 0:
            start = time.time()
            if i % 10 == 0:
                V, P = glasso(matreg(V), rho, 1)
            # V, P = glasso(V, rho, 1)
                # print('glasso time:', time.time() - start)
        gloss = np.sum(abs(P))
        # sparsify invV and return invV
        # update K using posterior of W
        k = diag(invXXK) + 1 / M * (YXinvXXK.T * solve(V, YXinvXXK).T).sum(-1)
        if lmd > 0:
            ksqrt = 1 + 8 * lmd * k
            k = (np.sqrt(ksqrt*(ksqrt>0))-1)/(4*lmd)
    invXXK = (identity(D) - X @ inv(identity(N) + X.T * k @ X) @ X.T * k).T * k
    W = Y @ X.T @ invXXK
    np.savez(f'{code}_i{i}.npz',V=V,P=P,k=k,W=W,lls=lls)
