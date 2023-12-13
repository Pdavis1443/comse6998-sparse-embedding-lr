import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math
from scipy.io import mmread
from scipy.sparse import csr_array, lil_array
from scipy.sparse.linalg import lsqr
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

def djl(X_train, X_test, y_train, y_test, k):
    tick = time.time()
    n = X_train.shape[0]
    tick2 = time.time()
    S = csr_array(np.random.normal(size = (k, n)) * (1.0 / math.sqrt(k)))
    SX = S @ X_train
    Sy = S @ y_train
    tock2 = time.time()
    lr, istop, itn, r1norm = lsqr(SX, Sy)[:4]
    tock = time.time()
    runtime = tock - tick
    time_reducing = tock2 - tick2

    print("DJL reduction (k = %d)" % k)
    print("\tMean absolute error: " + str(mean_absolute_error(y_test, X_test @ lr)))
    print("\tTime to complete (s): " + str(runtime))
    print("\tTime performing DJL reduction (s): " + str(time_reducing))
    print("\tIterations: " + str(itn))

def full_lr(X_train, X_test, y_train, y_test):
    tick = time.time()
    lr, istop, itn, r1norm = lsqr(X_train, y_train)[:4]
    tock = time.time()
    runtime = tock - tick

    print("Full linear regression:")
    print("\tMean absolute error: " + str(mean_absolute_error(y_test, X_test @ lr)))
    print("\tTime to complete (s): " + str(runtime))
    print("\tIterations: " + str(itn))

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
    print("\tMean absolute error: " + str(mean_absolute_error(y_test, X_test @ lr)))
    print("\tTime to complete (s): " + str(runtime))
    print("\tTime performing Sparse reduction (s): " + str(time_reducing))
    print("\tIterations: " + str(itn))

X, y = get_c41()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print("Data shapes: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

full_lr(X_train, X_test, y_train, y_test)
djl(X_train, X_test, y_train, y_test, 300)
sparse_embedding(X_train, X_test, y_train, y_test, 1000)





