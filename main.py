import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from scipy.io import mmread
from scipy.sparse import csr_array, lil_array, coo_array, vstack
from scipy.sparse.linalg import lsqr, spsolve
import random

def get_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston" 
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    bos = pd.DataFrame(data)

    bos['PRICE'] = target
    X = bos.drop('PRICE', axis=1).values
    y = bos.PRICE.values
    return (X, y)

def get_c41():
    X = mmread('c-41.mtx')
    y = mmread('c-41_b.mtx')
    return (X, y)


def full_lr(X_train, X_test, y_train, y_test):
    tick = time.time()
    lr, istop, itn, r1norm = lsqr(X_train, y_train)[:4]
    tock = time.time()
    runtime = tock - tick

    print("Full linear regression:")
    print("\tMean squared error (train): " + str(mean_squared_error(y_train, X_train @ lr)))
    print("\tMean squared error (test): " + str(mean_squared_error(y_test, X_test @ lr)))
    print("\tTime to complete (s): " + str(runtime))
    
    return lr, r1norm

def sparse_embedding(X_train, X_test, y_train, y_test, k):
    tick = time.time()
    n = X_train.shape[0]
    tick2 = time.time()
    S = lil_array((k, X_train.shape[0]))
    for j in range(n):
        i = random.randint(0, k - 1)
        a = 1 if random.randint(0, 1) == 1 else -1
        S[i, j] = a
    S = S.tocsr()
    SX = S @ X_train
    Sy = S @ y_train
    tock2 = time.time()
    lr, istop, itn, r1norm = lsqr(SX, Sy)[:4]
    tock = time.time()
    runtime = tock - tick
    time_reducing = tock2 - tick2

    print("Sparse embedding reduction (k = %d)" % k)
    print("\tMean squared error (train): " + str(mean_squared_error(y_train, X_train @ lr)))
    print("\tMean squared error (test): " + str(mean_squared_error(y_test, X_test @ lr)))
    print("\tTime to complete (s): " + str(runtime))
    print("\tTime performing Sparse reduction (s): " + str(time_reducing))
    
    return lr, r1norm
 
def random_subsample(X_train, X_test, y_train, y_test, k):
    tick = time.time()
    tick2 = time.time()
    SX, _, Sy, _ = train_test_split(X_train, y_train, train_size=k)
    tock2 = time.time()
    lr, istop, itn, r1norm = lsqr(SX, Sy)[:4]
    tock = time.time()
    runtime = tock - tick
    time_reducing = tock2 - tick2

    print("Random subsampling (k = %d)" % k)
    print("\tMean squared error (train): " + str(mean_squared_error(y_train, X_train @ lr)))
    print("\tMean squared error (test): " + str(mean_squared_error(y_test, X_test @ lr)))
    print("\tTime to complete (s): " + str(runtime))
    print("\tTime randomly subsampling (s): " + str(time_reducing))
    return lr, r1norm
 
 
 # Adds gaussian noise to some entries of X, ensuring that it will remain sparse if X is sparse.
def add_noise(X):
    n, m = X.shape
    E = lil_array((n ,m))
    total_err_nnzs = min(math.ceil(n * m / 1000), m, n)
    modified_indices = set()
    while len(modified_indices) < total_err_nnzs:
        r = random.randint(0, n - 1)
        c = random.randint(0, m - 1)
        if (r, c) not in modified_indices:
            modified_indices.add((r, c))
            E[r, c] = np.random.randn()
    return X + E.tocsr()
    
# Replicates the matrices X and y to some duplicity, adding a random Gaussian Noise matrix in the process
# The gaussian noise matrix is itself sparse, so each resulting replications will have no more than n*m/1000 additional
# non-zero entries over the original.
def replicate(X, y, reps, seed):
    random.seed(seed)
    yBig = np.vstack([y for i in range(reps)])
    Xs = [add_noise(X) for i in range(reps)]
    XBig = vstack(Xs)
    return (XBig, yBig)

def run(X, y, reps, k_sparse, k_random):
    X, y = replicate(X, y, reps, 1337)
    print("Running comparison for %d reps, %d sparse embedding rows, and %d random samples" % (reps, k_sparse, k_random) )
    print("X Shape (post reps): " + str(X.shape) + ", nnz: " + str(X.nnz))
    print("y Shape (post reps): " + str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    print("Data shapes (A_train, A_test, b_train, b_test): ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    lr, train_err = full_lr(X_train, X_test, y_train, y_test)
    print("||Ax-b||_2: " + str(train_err))
    #djl(X_train, X_test, y_train, y_test, 300)
    lr_sparse, train_err_sparse = sparse_embedding(X_train, X_test, y_train, y_test, k_sparse)
    print("||SAx - Sb||_2 for sparse embedding: " + str(train_err_sparse))
    lr_random, train_err_random = random_subsample(X_train, X_test, y_train, y_test, k_random)
    print("||SAx - Sb||_2 for random subsample: " + str(train_err_random))



X, y = get_c41()
reps = 5, 50, 500
k_sparses = [4000, 40_000, 400_000]
k_randoms = [4_000, 40_000, 400_000]
for (rep, k_sparse, k_random) in zip(reps, k_sparses, k_randoms):
    run(X, y, rep, k_sparse, k_random)

# def djl(X_train, X_test, y_train, y_test, k):
#     tick = time.time()
#     n = X_train.shape[0]
#     tick2 = time.time()
#     S = csr_array(np.random.normal(size = (k, n)) * (1.0 / math.sqrt(k)))
#     SX = S @ X_train
#     Sy = S @ y_train
#     tock2 = time.time()
#     lr, istop, itn, r1norm = lsqr(SX, Sy)[:4]
#     tock = time.time()
#     runtime = tock - tick
#     time_reducing = tock2 - tick2

#     print("DJL reduction (k = %d)" % k)
#     print("\tMean absolute error: " + str(mean_absolute_error(y_test, X_test @ lr)))
#     print("\tMean squared error: " + str(mean_squared_error(y_test, X_test @ lr)))
#     print("\tTime to complete (s): " + str(runtime))
#     print("\tTime performing DJL reduction (s): " + str(time_reducing))
#     print("\tIterations: " + str(itn))