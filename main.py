import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from scipy.io import mmread
from scipy.sparse import csr_array, lil_array, coo_array, vstack
from scipy.sparse.linalg import lsqr, spsolve
import random

def get_data(afile: str, bfile: str):
    """Reads in the specified A and b files as appropriate matrices/vectors (either sparse or dense, as appropriate)"""
    X = mmread(afile)
    y = mmread(bfile)
    return (X, y)


def full_lr(A, b):
    """Runs a linear regression on the full system (A,b), timing and reporting accuracy metrics"""
    tick = time.time()
    lr, istop, itn, r1norm = lsqr(A, b)[:4]
    tock = time.time()
    runtime = tock - tick

    Ax = A @ lr
    normAx = np.linalg.norm(Ax)
    print("Full linear regression:")
    print("\tMean squared error: " + str(mean_squared_error(b, Ax)))
    print("\t||Ax-b||_2(squared): " + str(r1norm) + " (%f)" % r1norm**2)
    print("\t||Ax||_2 (squared): " + str(normAx) + " (%f)" % normAx**2)
    print("\tTime to complete (s): " + str(runtime))
    
    return lr, r1norm

def sparse_embedding(A, b, k):
    """Generates a sparse embedding of size k x n, applies it to the system (A, b), and solves it. Returns the solution vector and the l2 norm of the solved Ax-b. Also reports timing and accuracy metrics."""
    tick = time.time()
    n = A.shape[0]
    tick2 = time.time()
    data = np.random.choice([1,-1], n)
    row_indices = np.random.choice(np.arange(k), n)
    col_indices = np.arange(n)
    S = coo_array((data, (row_indices, col_indices)), shape=(k, n))
    S = S.tocsr()
    tock2 = time.time()
    tick3 = time.time()
    SA = S @ A
    Sb = S @ b
    tock3 = time.time()
    lr, istop, itn, r1norm = lsqr(SA, Sb)[:4]
    tock = time.time()
    runtime = tock - tick
    time_generating = tock2 - tick2
    time_reducing = tock3 - tick3

    Ax = A @ lr
    normAx = np.linalg.norm(Ax)
    normAxmb = np.linalg.norm(Ax - b.reshape((b.shape[0],)))
    normSAx = np.linalg.norm(SA @ lr)
    print("Sparse embedding reduction (k = %d)" % k)
    print("\tMean squared error: " + str(mean_squared_error(b, Ax)))
    print("\t||Ax-b||_2(squared): " + str(normAxmb) + " (%f)" % normAxmb**2)
    print("\t||SAx-Sb||_2(squared): " + str(r1norm) + " (%f)" % r1norm**2)
    print("\t||Ax||_2 (squared): " + str(normAx) + " (%f)" % normAx**2)
    print("\t||SAx||_2 (squared): " + str(normSAx) + " (%f)" % normSAx**2)
    print("\tTime to complete (s): " + str(runtime))
    print("\tTime generating Sparse embedding matrix (s): " + str(time_generating))
    print("\tTime performing S*A and S*b (s): " + str(time_reducing))
    
    return lr, normAxmb
 
# def random_subsample(A, b, k):
#     """Randomly samples k rows of the system (A, b) and solves a linear regression on that system. Reports accuracy and runtime metrics."""
#     tick = time.time()
#     tick2 = time.time()
#     SA, _, Sb, _ = train_test_split(A, b, train_size=k)
#     tock2 = time.time()
#     lr, istop, itn, r1norm = lsqr(SA, Sb)[:4]
#     tock = time.time()
#     runtime = tock - tick
#     time_reducing = tock2 - tick2

#     Ax = A @ lr
#     normAxmb = np.linalg.norm(Ax - b.reshape((b.shape[0],)))
#     print("Random subsampling (k = %d)" % k)
#     print("\tMean squared error: " + str(mean_squared_error(b, Ax)))
#     print("\t||Ax-b||_2(squared): " + str(normAxmb) + " (%f)" % normAxmb**2)
#     print("\t||SAx-Sb||_2(squared): " + str(r1norm) + " (%f)" % r1norm**2)
#     print("\tTime to complete (s): " + str(runtime))
#     print("\tTime randomly subsampling (s): " + str(time_reducing))
#     return lr, normAxmb
 
def add_noise(X):
    """Adds gaussian noise to some entries of X, ensuring that it will remain sparse if X is sparse."""
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

def replicate(X, y, reps, seed):
    """
    Replicates the matrices X and y to some duplicity, adding a random Gaussian Noise matrix in the process
    The gaussian noise matrix is itself sparse, so each resulting replications will have no more than n*m/1000 additional
    non-zero entries over the original.
    """
    random.seed(seed)
    yBig = np.vstack([y for i in range(reps)])
    Xs = [add_noise(X) for i in range(reps)]
    XBig = vstack(Xs)
    return (XBig, yBig)

def run(A, b, reps, k_sparse):
    A, b = replicate(A, b, reps, 1337)
    print("Running comparison for %d reps and %d sparse embedding rows" % (reps, k_sparse) )
    print("A Shape (post reps): " + str(A.shape) + ", nnz: " + str(A.nnz))
    print("b Shape (post reps): " + str(b.shape))

    lr, train_err = full_lr(A, b)
    #djl(A, A, b, b, 300)
    lr_sparse, train_err_sparse = sparse_embedding(A, b, k_sparse)



A, b = get_data('illc1033.mtx', 'illc1033_b.mtx')
reps = [400, 4_000, 40_000, 80_000, 200_000] # Experiment 1
k_sparses = [4000] # Experiment 1
#reps = [4000] # Experiment 2
#k_sparses = [400, 800, 1600, 3200, 6400, 12_800, 25_600, 51_200] Experiment 2
for rep in reps:
    for k_sparse in k_sparses:
        run(A, b, rep, k_sparse)

# def djl(A, A, b, b, k):
#     """Creates a DJL transform of size kxn, applies it to the system A, then solves the linear regression. Reports accuracy and runtime metrics."""
#     tick = time.time()
#     n = A.shape[0]
#     tick2 = time.time()
#     S = csr_array(np.random.normal(size = (k, n)) * (1.0 / math.sqrt(k)))
#     SX = S @ A
#     Sy = S @ b
#     tock2 = time.time()
#     lr, istop, itn, r1norm = lsqr(SX, Sy)[:4]
#     tock = time.time()
#     runtime = tock - tick
#     time_reducing = tock2 - tick2

#     print("DJL reduction (k = %d)" % k)
#     print("\tMean absolute error: " + str(mean_absolute_error(b, A @ lr)))
#     print("\tMean squared error: " + str(mean_squared_error(b, A @ lr)))
#     print("\tTime to complete (s): " + str(runtime))
#     print("\tTime performing DJL reduction (s): " + str(time_reducing))
#     print("\tIterations: " + str(itn))