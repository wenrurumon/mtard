

'''
python hardi.py 0.1 0.925 0 0.1 1e-6 2 50 3 1
'''

if __name__ == '__main__':

    import os
    import sys

    alpha = float(sys.argv[1])
    lmd1 = float(sys.argv[2])  # Thresholding V
    lmd2 = float(sys.argv[3])  # Graphical Lasso
    lmd3 = float(sys.argv[4])  # PXEM
    tol = float(sys.argv[5])
    ncore = sys.argv[6]
    niter = int(sys.argv[7])
    datai = int(sys.argv[8])
    verbose = int(sys.argv[9])

    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore

    import numpy as np
    import dataseting
    import mblr
    import matplotlib.pyplot as plot

    X, Y, A, Z, xindex, yindex, aindex, zindex, age, female = dataseting.load_hexin0707()

    agetier = []
    agetier.append(age < 100)
    agetier.append(female)
    agetier.append(female == False)
    for i in range(4):
        agei = (age >= i * 10 + 20) & (age <= i * 10 + 30)
        agetier.append(agei)

    datasets = dataseting.split_dataset2(
        np.concatenate((X, Z[0, :].reshape(1, len(age))), axis=0),
        np.concatenate((Y, A), axis=0),
        agetier
    )

    X = datasets[datai][0]
    Y = datasets[datai][1]
    V, P, W, alpha, lls, para, model = mblr.hard(X, Y, niter, alpha, lmd1, lmd2, lmd3, tol, verbose)
    para.append(datai)
    np.savez(f'test/hard_{datai}_{lmd1}_{lmd2}_{lmd3}.npz',
             V=V,P=P,W=W,alpha=alpha,lls=lls,para=para,X=X,Y=Y)