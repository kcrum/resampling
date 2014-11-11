# Bootstrap Resampling

This follows an example from section 5.2 of "An Introduction to Statistical
Learning." Assume we invest in two assets 'X' and 'Y'. We invest a fraction of
our money 'f' in X and 1-f in Y. Minimizing Var[fX + (1-f)Y] as a function of
f, we find an estimate 'fhat' for f given by:
   fhat = (Var[X] - Cov[X,Y])/(Var[X] + Var[Y] - 2Cov[X,Y]).

Assume we have a sample of nevts=100 (x,y) pairs for some Var[X], Var[Y], and
Cov[X,Y]. Use the bootstrap technique to generate nsets=1000 bootstrap sets,
and estimate the distribution of fhat. Compare this to the distribution you get
by throwing 1000 sets from truth.  
