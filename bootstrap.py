import numpy as np
import matplotlib.pyplot as plt

# This follows an example from section 5.2 of "An Introduction to Statistical
# Learning." Assume we invest in two assets 'X' and 'Y'. We invest a fraction of
# our money 'f' in X and 1-f in Y. Minimizing Var[fX + (1-f)Y] as a function of 
# f, we find an estimate 'fhat' for f given by:
#    fhat = (Var[X] - Cov[X,Y])/(Var[X] + Var[Y] - 2Cov[X,Y]). 
#
# Assume we have a sample of nevts=100 (x,y) pairs for some Var[X], Var[Y], and 
# Cov[X,Y]. Use the bootstrap technique to generate nsets=1000 bootstrap sets, 
# and estimate the distribution of fhat. Compare this to the distribution you get
# by throwing 1000 sets from truth.

meanxtrue, meanytrue = 0, 0
varxtrue = 1
varytrue = 1.25
covxytrue = 0.5
truemeans = [meanxtrue, meanytrue]
truecovmat = [[varxtrue, covxytrue],[covxytrue, varytrue]]
nevts, nsets = 100, 1000

# Calculate fhat
def fhat(covmat):
    varx, vary, covxy = covmat[0][0], covmat[1][1], covmat[0][1]
    return (vary - covxy)/(varx + vary - 2*covxy)

# Return a data set pulled from a 2-d normal with 'size' entries.
def make_data(means, covmat, size):
    return np.random.multivariate_normal(means, covmat, size)

# Plot an (n x 2) array
def plot_set(dataset):
    plt.scatter(dataset[:,0], dataset[:,1])
    plt.show()

# Make bootstrap set with 'size' events.
def bootstrap_set(dataset, size):
    indices = np.random.choice(range(len(dataset)), size)
    bootset = []
    for i in indices:
        bootset.append(dataset[i])
    return bootset

# Make distributions of fhat for bootstrap approach and by pulling from 
# distribution. 
def fhat_distributions(bootdata):
    fhatboot = []
    fhatsim = []

    for i in xrange(nsets):
        # Get new set pulled from truth
        idata = make_data(truemeans, truecovmat, nevts)
        icovmat = np.cov(idata,rowvar=0)
        fhatsim.append(fhat(icovmat))

        # Get bootstrap set pulled from bootdata
        iBSdata = bootstrap_set(bootdata, len(bootdata))
        iBScovmat = np.cov(iBSdata,rowvar=0)
        fhatboot.append(fhat(iBScovmat))
    
    return np.array(fhatsim), np.array(fhatboot)


def main(verbose=True):
    fhattrue = fhat(truecovmat)

    # Pull the data set from which you will draw your bootstrap sets.
    bootdata = make_data(truemeans, truecovmat, nevts)
    bootcovmat = np.cov(bootdata,rowvar=0)

    fhatsim, fhatboot = fhat_distributions(bootdata)    

    # Output some statistics
    if verbose:
        print '-'*70
        print 'True fhat: ', fhattrue
        print 'From the bootstrap parent set: '
        print '   fhat: ', fhat(bootcovmat)
        print '   Cov. mat.: ', bootcovmat
        print 'Simulated fhat mean:', fhatsim.mean(), ' Std. dev.: ', \
            fhatsim.std(ddof=1)
        print 'Bootstrap fhat mean:', fhatboot.mean(), ' Std. dev.: ', \
            fhatboot.std(ddof=1)
        print '-'*70

    # Plot
    fig = plt.figure()

    ax1 = fig.add_subplot(1,2,1)    
    conts, edges, _ = ax1.hist(fhatsim, bins=20)
    ymax = 1.15*conts.max()
    ax1.plot([fhattrue,fhattrue], [0,ymax], 'r-', lw=2)
    ax1.set_ylim([0,ymax])
    ax1.set_title(r"$\hat{f}$ from 1,000 simulations")

    ax2 = fig.add_subplot(1,2,2)    
    ax2.hist(fhatboot, bins=edges)
    ax2.plot([fhattrue,fhattrue], [0,ymax], 'r-', lw=2)
    ax2.set_ylim([0,ymax])
    ax2.set_title(r"$\hat{f}$ from 1,000 bootstrap samples")

    plt.show()

if __name__ == '__main__':
    main()
    
