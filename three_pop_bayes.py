from __future__ import print_function
import dadi
import OutOfAfrica
import GPy, GPyOpt
import pylab
import scipy

from dadi import Numerics, PhiManip, Integration, Spectrum

# Numpy is the numerical library dadi is built upon
from numpy import array, atleast_2d, log, exp
from numpy.random import uniform
from scipy.optimize.slsqp import wrap_function

data = dadi.Spectrum.from_file('YRI.CEU.CHB.fs')
ns = data.sample_sizes

pts_l = [40,50,60]

# Now let's optimize parameters for this model.

# Parameters are: (nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs,
#                  mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs)
upper_bound = [100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 3, 3, 3]
lower_bound = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0, 0, 0, 0, 0, 0, 0]

# This is our initial guess for the parameters, which is somewhat arbitrary.
#p0 = [2, 0.1, 2, 2, 0.1, 2, 1, 1, 1, 1, 0.2, 0.2, 0.2]
# Make the extrapolating version of our demographic model function.
func = OutOfAfrica.OutOfAfrica
#func_ex_log = dadi.Numerics.make_extrap_log_func(func)
func_ex = dadi.Numerics.make_extrap_func(func)

# Perturb our parameters before optimization. This does so by taking each
# parameter a up to a factor of two up or down.
#p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
#                              lower_bound=lower_bound)

p0 = [uniform(l, u) for l, u in zip(lower_bound, upper_bound)]


print('Beginning optimization ************************************************')

BayesInference = dadi.Inference


def optimize_bayes(p0, data, model_func, pts, lower_bound=None, upper_bound=None,
                 verbose=0, flush_delay=0.5, epsilon=1e-3,
                 gtol=1e-5, multinom=False, maxiter=None, full_output=False,
                 func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                 output_file=None):

    def f_wrapped(x):
        print(x)
        return BayesInference._object_func_log(x.tolist()[0], *args)

    def f_obj_wrapped(x):
        return BayesInference._object_func(x.tolist()[0], *args)

    if output_file:
        output_stream = file(output_file, 'w')
    else:
        output_stream = BayesInference.sys.stdout

    args = (data, model_func, pts, lower_bound, upper_bound, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params,
            ll_scale, output_stream)

    p0 = BayesInference._project_params_down(p0, fixed_params)

    bounds = [{'domain': (l, u)} for l, u in zip(lower_bound, upper_bound)]

    myProblem = GPyOpt.methods.BayesianOptimization(f_obj_wrapped,
                                                    X=atleast_2d(p0),
                                                    domain=bounds,
                                                    acquisition_type='MPI',
                                                    verbosity=True,
                                                    maximize=False,
                                                    #maxiter=50,
                                                    num_cores=8,
                                                    maxiter=1
                                                    )

    myProblem.run_optimization(maxiter, verbosity=True)

    #myProblem.plot_convergence()
    print(myProblem.get_evaluations())
    #xopt = BayesInference._project_params_up(exp(myProblem.x_opt), fixed_params)
    xopt = BayesInference._project_params_up(myProblem.x_opt, fixed_params)

    # outputs = scipy.optimize.fmin_bfgs(BayesInference._object_func, xopt,
    #                                    epsilon=epsilon,
    #                                    args=args, gtol=gtol,
    #                                    full_output=True,
    #                                    disp=False,
    #                                    maxiter=maxiter)
    # xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    # xopt = BayesInference._project_params_up(xopt, fixed_params)


    return xopt


BayesInference.optimize_log = optimize_bayes
print('MONKEY PATCHED BAYES START')
popt = BayesInference.optimize_log(p0, data, func_ex, pts_l,
                                    lower_bound=lower_bound,
                                    upper_bound=upper_bound,
                                    verbose=len(p0), maxiter=10)
print(popt)
print('MONKEY PATCHED BAYES END')

#raise SystemExit

# The verbose argument controls how often progress of the optimizer should be
# printed. It's useful to keep track of optimization process.
print('Finshed optimization **************************************************')

# Calculate the best-fit model AFS.
model = func_ex(popt, ns, pts_l)
# Likelihood of the data given the model AFS.
ll_model = dadi.Inference.ll_multinom(model, data)
print('Maximum log composite likelihood: {0}'.format(ll_model))
# The optimal value of theta given the model.
theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))


fig = pylab.figure(1)
fig.clear()
dadi.Plotting.plot_1d_comp_multinom(model.marginalize([1,2]),
                                    data.marginalize([1,2]))
fig.savefig('gpyopt_1d_comp.png')

fig = pylab.figure(2)
fig.clear()
dadi.Plotting.plot_single_2d_sfs(data.marginalize([2]), vmin=1)
fig.savefig('gpyopt_2d_single.png')

fig = pylab.figure(3)
fig.clear()
dadi.Plotting.plot_2d_comp_multinom(model.marginalize([2]),
                                    data.marginalize([2]), vmin=1)
fig.savefig('gpyopt_2d_comp.png')

fig = pylab.figure(4, figsize=(8,10))
fig.clear()
dadi.Plotting.plot_3d_comp_multinom(model, data, vmin=1)
fig.savefig('gpyopt_3d_comp.png')

pylab.show()