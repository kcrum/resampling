import numpy as np
import matplotlib.pyplot as plt

# This follows an example from section 5.2 of "An Introduction to Statistical
# Learning." Assume we invest in two assets 'X' and 'Y'. We invest a fraction of
# our money 'f' in X and 1-f in Y. Minimizing Var[fX + (1-f)Y] as a function of 
# f, we find an estimate 'fhat' for f given by:
#    fhat = (Var[X] - Cov[X,Y])/(Var[X] + Var[Y] - 2Cov[X,Y]). 
#
# Assume we have a sample of n=100 (x,y) pairs for some Var[X], Var[Y], and 
# Cov[X,Y]. Use the bootstrap technique to generate m=1000 bootstrap sets, and 
# estimate the distribution of fhat. Compare this to the distribution you get
# by throwing 1000 sets from truth.

meanxtrue, meanytrue = 0, 0
varxtrue = 1
varytrue = 1.25
covxytrue = 0.5
truef = fhat(varx, vary, covxy)

n, m = 100, 1000

def fhat(varx, vary, covxy):
    return (vary - covxy)/(varx + vary - 2*covxy)

def make_data(meanx, meany, varx, vary, covxy, setsize):
    covmat = [[varx,covxy],
              [covxy,vary]]
    return np.random.multivariate_normal([meanx, meany], covmat, setsize)

def plot_set(dataset):
    plt.scatter(norm2d[:,0], norm2d[:,1])
    plt.show()

# Make bootstrap set with 'size' events.
def bootstrap_set(dataset, size=len(dataset)):
    indices = np.random.choice(range(len(dataset)), size)
    bootset = []
    for i in indices:
        bootset.append(dataset[i])
    return bootset
