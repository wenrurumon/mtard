import ctypes as ct
fortlib = ct.CDLL('./glassofast.so')
f = fortlib.glassofast
import numpy as np

def fglasso(S,alpha,niter):
    N = S.shape[0]
    S_ptr = S.ctypes.data_as(ct.POINTER(ct.c_double))
    L = alpha * np.ones((N,N), order='F')
    L_ptr = L.ctypes.data_as(ct.POINTER(ct.c_double))
    X = np.zeros([N,N], order='F')
    X_ptr = X.ctypes.data_as(ct.POINTER(ct.c_double))
    W = np.zeros([N,N], order='F')
    W_ptr = W.ctypes.data_as(ct.POINTER(ct.c_double))
    f(ct.c_int(N), S_ptr, L_ptr, ct.c_double(1e-12), ct.c_int(niter), ct.c_int(niter), ct.c_int(0), X_ptr, W_ptr)
    return(W,X)
