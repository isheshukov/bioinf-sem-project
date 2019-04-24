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

print('Beginning optimization ************************************************')

popt = BayesOpt.optimize(p0, data, func,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         verbose=1,
                         maxiter=4)
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


fig = pylab.figure(1)
fig.clear()
moments.Plotting.plot_1d_comp_multinom(model.marginalize([1, 2]),
                                       data.marginalize([1, 2]))
fig.savefig('plots/gpyopt_1d_comp.png')

fig = pylab.figure(2)
fig.clear()
moments.Plotting.plot_single_2d_sfs(data.marginalize([2]), vmin=1)
fig.savefig('plots/gpyopt_2d_single.png')

fig = pylab.figure(3)
fig.clear()
moments.Plotting.plot_2d_comp_multinom(model.marginalize([2]),
                                       data.marginalize([2]), vmin=1)
fig.savefig('plots/gpyopt_2d_comp.png')

fig = pylab.figure(4, figsize=(8, 10))
fig.clear()
moments.Plotting.plot_3d_comp_multinom(model, data, vmin=1)
fig.savefig('plots/gpyopt_3d_comp.png')