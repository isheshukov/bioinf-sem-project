from __future__ import print_function

import moments
from numpy.random import uniform
import lib.BayesOpt as BayesOpt
import pylab

from lib.OutOfAfrica_moments import OutOfAfrica


data = moments.Spectrum.from_file('lib/YRI.CEU.CHB.fs')
ns = data.sample_sizes
pts_l = [40, 50, 60]

# Parameters are: (nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs,
#                  mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs)
upper_bound = [100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 3, 3, 3]
lower_bound = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

func = OutOfAfrica
p0 = [uniform(l, u) for l, u in zip(lower_bound, upper_bound)]

# Fixing first two params
popt1 = [None, None, 0.129, 3.74, 0.070, 7.29, 3.65, 0.44, 0.28, 1.4, 0.211, 0.338, 0.058]
print('Beginning optimization ************************************************')

popt = BayesOpt.optimize(p0, data, func,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         verbose=1,
                         maxiter=100,
                         fixed_params=popt1)
print(popt)

print('Finshed optimization **************************************************')

# Calculate the best-fit model AFS.
model = func(popt, ns)
# Likelihood of the data given the model AFS.
ll_model = moments.Inference.ll_multinom(model, data)
print('Maximum log composite likelihood: {0}'.format(ll_model))
# The optimal value of theta given the model.
theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))