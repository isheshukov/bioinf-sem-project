import os
from time import time
import numpy as np
import moments
import GPyOpt
import scipy

BayesInference = moments.Inference

cntr = 0
best_result = np.inf
time_start = time()

# Optimization routine used in all of the scripts

def optimize(p0, data, model_func, lower_bound=None, upper_bound=None,
             verbose=0, flush_delay=0.5, epsilon=1e-3,
             gtol=1e-5, multinom=True, maxiter=1000, full_output=False,
             func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
             output_file=None, output_dir=None, log_params=False, **kwargs):

    # passing all parameters to objective function,
    # because gpyopt can't do it itself

    # Function wrapping our objective function
    #
    # It is needed, because we need to log
    # each evaluation of said function
    # and and exponent our parameters is flag `log_params`
    # is set.

    def f_obj_wrapped(x):
        global cntr
        global best_result
        global time_start

        cntr += 1
        print('f_obj_wrapped is being launched for the %d\'th time' % cntr)
        # return 1000
        f_args = x.tolist()[0]
        if log_params:
            print('Moments parameters BEFORE exponentiating:')
        else:
            print('Moments parameters:')
        # print(f_args)
        print(', '.join('%0.3f' % one_arg for one_arg in f_args))
        if log_params:
            f_args = np.exp(f_args)
            print('Moments parameters after exponentiating:')
            # print(f_args)
            print(', '.join('%0.3f' % one_arg for one_arg in f_args))
        # print(args)
        result = moments.Inference._object_func(f_args, *args)
        print('Moments returned:')
        # print(result)
        print('%0.3f' % result)
        print('')

        if result < best_result:
            best_result = result

        log_file.write('%d %0.3f %0.3f %s %f \n' % (cntr,
                result,
                best_result,
                np.array2string(f_args, formatter={'float_kind': lambda x: "%.3f" % x}),
                time() - time_start))

        return result

    # Creating output dir for containing results

    _out_dir = ("out/" + output_dir + "/") if output_dir is not None else ""

    if output_file:
        output_stream = open(output_file, 'w')
    else:
        output_stream = BayesInference.sys.stdout

    log_file_path = _out_dir + "log.txt"

    if not os.path.exists(os.path.dirname(log_file_path)):
        try:
            os.makedirs(os.path.dirname(log_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if output_dir:
        log_file = open(log_file_path, 'w', 1)
    log_file.write('Iteration, Objective function, Best objective function, Parameters, Time elapsed (s)\n')


    # Optimization has problems when it tries to evaluate a point 
    # directly on the bounds of search space.
    # That way we reduce the instability by limting possible evaluations.

    EPS = 1e-6
    lower_bound_args = map(lambda x: (x - EPS) if ((x - EPS) > 0.0) else 0.0, lower_bound)
    upper_bound_args = map(lambda x: (x + EPS), upper_bound)
    args = (data, model_func, lower_bound_args, upper_bound_args, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params,
            ll_scale, output_stream)

    # Fixing parameters.
    # Necessary if we try to optimize only a subset of already known parameters
    # We used it for testing, when our optimization was producing bad results.

    if isinstance(p0[0], list):
        p0 = [BayesInference._project_params_down(cur_p0, fixed_params) for cur_p0 in p0]
    else:
        p0 = BayesInference._project_params_down(p0, fixed_params)


    # Creating bounds dict for GPyOpt to optimize on.

    if fixed_params is None:
        bounds = [{'domain': (l, u)} for l, u in zip(lower_bound, upper_bound)]
    else:
        bounds = [{'domain': (l, u)} for l, u, p in zip(lower_bound, upper_bound, fixed_params) if p is None]

    # Logarithmic parameters are somehow producing better results
    # and optimization converges faster if this flag is used.
    # This may be happening because optimal point is often 
    # somewhere near the point (1,1,1,1...,1)

    if log_params:
        # print bounds
        for cur_dict in bounds:
            cur_min, cur_max = cur_dict['domain']
            cur_dict['domain'] = (np.log(cur_min), np.log(cur_max))
        # print bounds

        # print p0
        p0 = np.log(p0)
        # print p0

    # Calling GPyOpt for optimization
    myProblem = GPyOpt.methods.BayesianOptimization(f_obj_wrapped,
                                                    X=np.atleast_2d(p0), # ensuring that the starting point of optimization is 2d array
                                                                         # because GPyOpt works only on such points
                                                    domain=bounds,       # search space
#                                                    acquisition_type='MPI',
                                                    acquisition_type='EI', # Use `Expected Improvement` acquisition function
                                                    verbosity=True, # print logging info
                                                    maximize=False, # minimizing function
#                                                    num_cores=1,
                                                    num_cores=10, 
                                                    verbosity_model=True,
                                                    **kwargs
                                                    )

    myProblem.run_optimization(maxiter, verbosity=True)

    # Optimization ended. Writing results.

    if not os.path.exists(_out_dir + "plots/"):
        try:
            os.makedirs(_out_dir + "plots/")
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if not os.path.exists(_out_dir + "reports/"):
        try:
            os.makedirs(_out_dir + "reports/")
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    myProblem.plot_acquisition(_out_dir + "plots/acquisition.png")
    myProblem.plot_convergence(_out_dir + "plots/convergence.png")
    myProblem.save_evaluations(_out_dir + "reports/evals.tsv")
    myProblem.save_report(_out_dir + "reports/report.txt")

    xopt = BayesInference._project_params_up(myProblem.x_opt, fixed_params)
    xopt = BayesInference._project_params_down(xopt, fixed_params)

    # outputs = scipy.optimize.fmin_bfgs(BayesInference._object_func, xopt,
    #                                    epsilon=epsilon,
    #                                    args=args, gtol=gtol,
    #                                    full_output=True,
    #                                    disp=False,
    #                                    maxiter=maxiter,
    #                                    fixed_params=fixed_params)
    # xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = BayesInference._project_params_up(xopt, fixed_params)

    log_file.close()

    return xopt
