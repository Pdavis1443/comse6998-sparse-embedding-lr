# Sparse Embedding Linear Regression

This repository was developed to host code for a project in my "Algorithms for Massive Data" class at Columbia University. It holds a single `main.py` which will, when executed, run a series of linear regressions on an expanded version of the [illc1033 data set](http://sparse.tamu.edu/HB/illc1033), taken from the SuiteSparse Matrix collection.

The runs are comparative in nature, assessing both a full linear regression as well as a sparse embedding matrix applied to the linear system before regression. 

There's also some commented out code for Distributive Johnson-Lindenstrauss transforms and random subsampling, which I decided to omit from my final writeup, but left commented out for reference.

# Dependencies
Be sure to install the `requirements.txt` before running `main.py`. This project was made for python 3 and needs:
- scipy
- numpy
- scikit-learn
- pandas