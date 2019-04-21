from __future__ import print_function
import moments 
from OutOfAfrica_moments import OutOfAfrica
import GPy, GPyOpt
import pylab
import scipy
import matplotlib.pyplot as plt

# Numpy is the numerical library dadi is built upon
from numpy import array, atleast_2d, log, exp
from numpy.random import uniform
from numpy import pad
from scipy.optimize.slsqp import wrap_function

from itertools import islice

data = moments.Spectrum.from_file('YRI.CEU.CHB.fs')
ns = data.sample_sizes

pts_l = [40,50,60]

# Now let's optimize parameters for this model.

# Parameters are: (nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs,
#                  mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs)
upper_bound = [100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 3, 3, 3]
lower_bound = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

# This is our initial guess for the parameters, which is somewhat arbitrary.
#p0 = [2, 0.1, 2, 2, 0.1, 2, 1, 1, 1, 1, 0.2, 0.2, 0.2]
# Make the extrapolating version of our demographic model function.
func = OutOfAfrica
#func_ex_log = dadi.Numerics.make_extrap_log_func(func)
#func_ex = moments.Numerics.make_extrap_func(func)

# Perturb our parameters before optimization. This does so by taking each
# parameter a up to a factor of two up or down.
#p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
#                              lower_bound=lower_bound)

p0 = [uniform(l, u) for l, u in zip(lower_bound, upper_bound)]


print('Beginning optimization ************************************************')

BayesInference = moments.Inference


def optimize_bayes(p0, data, model_func, lower_bound=None, upper_bound=None,
                 verbose=0, flush_delay=0.5, epsilon=1e-3,
                 gtol=1e-5, multinom=True, maxiter=None, full_output=False,
                 func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                 output_file=None):

    def f_obj_wrapped(x):
        return BayesInference._object_func(x.tolist()[0], *args)

    if output_file:
        output_stream = file(output_file, 'w')
    else:
        output_stream = BayesInference.sys.stdout

    args = (data, model_func, lower_bound, upper_bound, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params,
            ll_scale, output_stream)

    p0 = BayesInference._project_params_down(p0, fixed_params)

    bounds = [{'domain': (l, u)} for l, u in zip(lower_bound, upper_bound)]

    #myProblem = GPyOpt.methods.BayesianOptimization(f_obj_wrapped,
    #                                                X=atleast_2d(p0),
    #                                                domain=bounds,
    #                                                acquisition_type='MPI',
    #                                                verbosity=True,
    #                                                maximize=False,
    #                                                #maxiter=50,
    #                                                num_cores=8,
    #                                                #normalize_Y=True
    #                                                )

    #myProblem.run_optimization(maxiter, verbosity=True)
    #myProblem.save_evaluations("evals.csv")
    #myProblem.save_report("report.csv")

    #eval_X, eval_Y = myProblem.get_evaluations()

    ###
    ### ACQUISITION PROJECTION PLOTS
    ###
    model_parameters = ["nuAf", "nuB", "nuEu0", "nuEu", "nuAs0", "nuAs", "mAfB", "mAfEu", "mAfAs", "mEuAs", "TAf", "TB",
                        "TEuAs"]

    tot = len(model_parameters)
    cols = 3
    rows = tot // cols
    rows += tot % cols
    position = range(1, tot + 1)

    # fig, axs = plt.subplots(rows, cols, figsize=(16,16), sharex=True)
    #
    # for k, (ax, X, variable) in enumerate(zip(axs.flat, eval_X.T, model_parameters)):
    #     if k < tot:
    #         ax.plot(range(len(X)), abs(X - pad(X, (1,0), mode='constant')[: -1]), 'ro-')
    #         ax.set_title(variable)
    #     else:
    #         fig.delaxes(ax)
    #
    # fig.text(0.5, 0.01, "Iteration", ha="center", va="center")
    # fig.text(0.01, 0.5, "|X_n - X_(n-1)|", ha="center", va="center", rotation=90)
    #
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig("projections.pdf")

    #myProblem.plot_convergence("moments_convergence.pdf")
    #xopt = BayesInference._project_params_up(exp(myProblem.x_opt), fixed_params)
    #xopt = BayesInference._project_params_up(myProblem.x_opt, fixed_params)

    xopt = [1.00000000e-03, 1.00000000e+02, 1.00000000e-03, 1.00000000e-03,
 6.37305388e+01, 1.00000000e+02, 1.00000000e+01, 1.00000000e-04,
 1.00000000e-04, 1.00000000e+01, 3.00000000e+00, 1.00000000e-04,
 1.00000000e-04]

    outputs = scipy.optimize.fmin_bfgs(BayesInference._object_func, xopt,
                                        epsilon=epsilon,
                                        args=args, gtol=gtol,
                                        full_output=True,
                                        disp=False,
                                        maxiter=maxiter)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = BayesInference._project_params_up(xopt, fixed_params)


    return xopt


BayesInference.optimize = optimize_bayes
print('MONKEY PATCHED BAYES START')
popt = BayesInference.optimize(p0, data, func,
                                    lower_bound=lower_bound,
                                    upper_bound=upper_bound,
                                    verbose=len(p0), maxiter=100)
print(popt)
print('MONKEY PATCHED BAYES END')

#raise SystemExit

# The verbose argument controls how often progress of the optimizer should be
# printed. It's useful to keep track of optimization process.
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
moments.Plotting.plot_1d_comp_multinom(model.marginalize([1,2]),
                                    data.marginalize([1,2]))
fig.savefig('gpyopt_1d_comp.png')

fig = pylab.figure(2)
fig.clear()
moments.Plotting.plot_single_2d_sfs(data.marginalize([2]), vmin=1)
fig.savefig('gpyopt_2d_single.png')

fig = pylab.figure(3)
fig.clear()
moments.Plotting.plot_2d_comp_multinom(model.marginalize([2]),
                                    data.marginalize([2]), vmin=1)
fig.savefig('gpyopt_2d_comp.png')

fig = pylab.figure(4, figsize=(8,10))
fig.clear()
moments.Plotting.plot_3d_comp_multinom(model, data, vmin=1)
fig.savefig('gpyopt_3d_comp.png')

#pylab.show()



# # These are the actual best-fit model parameters, which we found through
# # longer optimizations and confirmed by running multiple optimizations.
# # We'll work with them through the rest of this script.
# #popt = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
# print('Best-fit parameters: {0}'.format(popt))
#
# # Calculate the best-fit model AFS.
# model = func_ex(popt, ns, pts_l)
# # Likelihood of the data given the model AFS.
# ll_model = dadi.Inference.ll_multinom(model, data)
# print('Maximum log composite likelihood: {0}'.format(ll_model))
# # The optimal value of theta given the model.
# theta = dadi.Inference.optimal_sfs_scaling(model, data)
# print('Optimal value of theta: {0}'.format(theta))
#
# # Plot a comparison of the resulting fs with the data.
# import pylab
# pylab.figure(1)
# dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=1, resid_range=3,
#                                     pop_ids =('YRI','CEU'))
# # This ensures that the figure pops up. It may be unecessary if you are using
# # ipython.
# pylab.show()
# # Save the figure
# pylab.savefig('YRI_CEU_bayes.png', dpi=50)
#
# # Let's generate some data using ms, if you have it installed.
# mscore = demographic_models.prior_onegrow_mig_mscore(popt)
# # I find that it's most efficient to simulate with theta=1, average over many
# # iterations, and then scale up.
# mscommand = dadi.Misc.ms_command(1., ns, mscore, int(1e5))
# # If you have ms installed, uncomment these lines to see the results.
#
# # We use Python's os module to call this command from within the script.
# import os
# return_code = os.system('{0} > test.msout'.format(mscommand))
# # We check the return code, so the script doesn't crash if you don't have ms
# # installed
# if return_code == 0:
#     msdata = dadi.Spectrum.from_ms_file('test.msout')
#     pylab.figure(2)
#     dadi.Plotting.plot_2d_comp_multinom(model, theta*msdata, vmin=1,
#                                         pop_ids=('YRI','CEU'))
#     pylab.show()
#
# # Estimate parameter uncertainties using the Godambe Information Matrix, to
# # account for linkage in the data. To use the GIM approach, we need to have
# # spectra from bootstrapping our data.  Let's load the ones we've provided for
# # the example.
# # (We're using Python list comprehension syntax to do this in one line.)
# all_boot = [dadi.Spectrum.from_file('bootstraps/{0:02d}.fs'.format(ii))
#             for ii in range(100)]
# uncerts = dadi.Godambe.GIM_uncert(func_ex, pts_l, all_boot, popt, data,
#                                   multinom=True)
# # uncert contains the estimated standard deviations of each parameter, with
# # theta as the final entry in the list.
# print('Estimated parameter standard deviations from GIM: {0}'.format(uncerts))
#
# # For comparison, we can estimate uncertainties with the Fisher Information
# # Matrix, which doesn't account for linkage in the data and thus underestimates
# # uncertainty. (Although it's a fine approach if you think your data is truly
# # unlinked.)
# uncerts_fim = dadi.Godambe.FIM_uncert(func_ex, pts_l, popt, data, multinom=True)
# print('Estimated parameter standard deviations from FIM: {0}'.format(uncerts_fim))
#
# print('Factors by which FIM underestimates parameter uncertainties: {0}'.format(uncerts/uncerts_fim))
#
# # What if we fold the data?
# # These are the optimal parameters when the spectrum is folded. They can be
# # found simply by passing data.fold() to the above call to optimize_log.
# popt_fold =  array([1.907,  0.073,  1.830,  0.899,  0.425,  0.113])
# uncerts_folded = dadi.Godambe.GIM_uncert(func_ex, pts_l, all_boot, popt_fold,
#                                          data.fold(), multinom=True)
# print('Folding increases parameter uncertainties by factors of: {0}'.format(uncerts_folded/uncerts))
#
# # Let's do a likelihood-ratio test comparing models with and without migration.
# # The no migration model is implemented as
# # demographic_models.prior_onegrow_nomig
# func_nomig = demographic_models.prior_onegrow_nomig
# func_ex_nomig = dadi.Numerics.make_extrap_log_func(func_nomig)
# # These are the best-fit parameters, which we found by multiple optimizations
# popt_nomig = array([ 1.897,  0.0388,  9.677,  0.395,  0.070])
# model_nomig = func_ex_nomig(popt_nomig, ns, pts_l)
# ll_nomig = dadi.Inference.ll_multinom(model_nomig, data)
#
# # Since LRT evaluates the complex model using the best-fit parameters from the
# # simple model, we need to create list of parameters for the complex model
# # using the simple (no-mig) best-fit params.  Since evalution is done with more
# # complex model, need to insert zero migration value at corresponding migration
# # parameter index in complex model. And we need to tell the LRT adjust function
# # that the 3rd parameter (counting from 0) is the nested one.
# p_lrt = [1.897,  0.0388,  9.677, 0, 0.395,  0.070]
#
# adj = dadi.Godambe.LRT_adjust(func_ex, pts_l, all_boot, p_lrt, data,
#                               nested_indices=[3], multinom=True)
# D_adj = adj*2*(ll_model - ll_nomig)
# print('Adjusted D statistic: {0:.4f}'.format(D_adj))
#
# # Because this is test of a parameter on the boundary of parameter space
# # (m cannot be less than zero), our null distribution is an even proportion
# # of chi^2 distributions with 0 and 1 d.o.f. To evaluate the p-value, we use the
# # point percent function for a weighted sum of chi^2 dists.
# pval = dadi.Godambe.sum_chi2_ppf(D_adj, weights=(0.5,0.5))
# print('p-value for rejecting no-migration model: {0:.4f}'.format(pval))
