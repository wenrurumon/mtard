
import os
import pandas as pd
import numpy as np
import random
import sys
import csv

#ncore = '8'
#os.environ["MKL_NUM_THREADS"] = ncore
#os.environ["OMP_NUM_THREADS"] = ncore
#os.environ["NUMEXPR_NUM_THREADS"] = ncore
#os.environ["OPENBLAS_NUM_THREADS"] = ncore

from mblr_hxx import *

################################################################
# Modeling
################################################################


# Random data set X and Y with 1000 Sample，100Y，300X

# Subset 100，300，500，1000 Samples do：set.seed(1:10)
    #2. MTARD(Y~X) -> likehood for the total system
    #3. Linear Model: Yi ~ X for each Yi -> likehood
    #4. Struc Equa: Yi ~ Y[-i] + X for each Yi -> likehood
    #5. #3#4 with ARD Regression

# {seed,sample,method,likehood}


# Generate ====

d = 100
m = 300
N = 1100

Y,W,X,V = dummy(d=d, m=m, N=N, seed=1, lambda1=0.9, lambda2=0.8) # X: m*N; Y: d*N
#Y,W,X,V = dummy(d=100, m=500, N=1100, seed=1, lambda1=0.9, lambda2=0.8) # X: m*N; Y: d*N


# Split & Sample ====
test_size = 100
X_train, X_test, Y_train, Y_test = split_XY(X, Y, test_size=test_size, seed=1)

size = [200,400,600,800,1000]
seed = [i for i in range(10)]
datasets = sample_XY(X_train, Y_train, size=size, seed=seed)


# Compute ll ====
from scipy.stats import norm

sim_rlt = []
for i in range(len(size)):
    for j in range(len(seed)):

        size_tmp = size[i]
        seed_tmp = seed[j]
        X_tmp = datasets[i][j][0]
        Y_tmp = datasets[i][j][1]

        lr_W = modeling(X_tmp, Y_tmp, mode='lr')
        ard_W = modeling(X_tmp, Y_tmp, mode='ard')
        strclr_W = modeling(X_tmp, Y_tmp, mode='strclr')
        strcard_W = modeling(X_tmp, Y_tmp, mode='strcard')
        mtard_W = modeling(X_tmp, Y_tmp, mode='mtard')

        lr_mse = calc_mse(X_test, Y_test, lr_W)
        ard_mse = calc_mse(X_test, Y_test, ard_W)
        strclr_mse = calc_strc_mse(X_test.T, Y_test.T, strclr_W)
        strcard_mse = calc_strc_mse(X_test.T, Y_test.T, strcard_W)
        mtard_mse = calc_mse(X_test, Y_test, mtard_W)

        if m < size_tmp:
            lr_ll = calc_ll(X_test, Y_test, lr_W)
            ard_ll = calc_ll(X_test, Y_test, ard_W)
            strclr_ll = calc_strc_ll(X_test.T, Y_test.T, strclr_W)
            strcard_ll = calc_strc_ll(X_test.T, Y_test.T, strcard_W)
            mtard_ll = calc_ll(X_test, Y_test, mtard_W)
        else:
            lr_ll = ard_ll = strclr_ll = strcard_ll = mtard_ll = 0

        sim_rlt.append((size_tmp, seed_tmp,
                        lr_ll, ard_ll, strclr_ll, strcard_ll, mtard_ll,
                        lr_ll/test_size, ard_ll/test_size, strclr_ll/test_size, strcard_ll/test_size, mtard_ll/test_size,
                        lr_mse, ard_mse, strclr_mse, strcard_mse, mtard_mse))

        print("size: ", size_tmp, "; seed: ", seed_tmp)

        if size_tmp == 1000:
            break


# Save ====
csv_file = "/Users/huxiaoxi/Documents/Fudan/ARD/Data/Simulation/sim_rlt.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['sample size', 'seed',
                        'Linear LL', 'ARD LL', 'Strc Linear LL', 'Strc ARD LL', 'MTARD LL',
                        'Linear A-LL', 'ARD A-LL', 'Strc Linear A-LL', 'Strc ARD A-LL', 'MTARD A-LL',
                        'Linear MSE', 'ARD MSE', 'Strc Linear MSE', 'Strc ARD MSE', 'MTARD MSE'])
    for row in sim_rlt:
        writer.writerow(row)


## Mean ====
grouped_data = {}
for item in sim_rlt:
    key = item[0]
    value = item[2:]
    if key in grouped_data:
        grouped_data[key].append(value)
    else:
        grouped_data[key] = [value]

averages = [(key, [statistics.mean(sublist) for sublist in zip(*values)]) for key, values in grouped_data.items()]

