import moments
import numpy as np
import GPyOpt

BayesInference = moments.Inference


def optimize(p0, data, model_func, lower_bound=None, upper_bound=None,
             verbose=0, flush_delay=0.5, epsilon=1e-3,
             gtol=1e-5, multinom=True, maxiter=0, full_output=False,
             func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
             output_file=None):

    # passing all parameters to objective function, because gpyopt can't do it itself
    def f_obj_wrapped(x):
        return moments.Inference._object_func(x.tolist()[0], *args)

    if output_file:
        output_stream = file(output_file, 'w')
    else:
        output_stream = BayesInference.sys.stdout

    args = (data, model_func, lower_bound, upper_bound, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params,
            ll_scale, output_stream)

    # Fixing parameters. Not sure if will work with gpy
    p0 = BayesInference._project_params_down(p0, fixed_params)

    bounds = [{'domain': (l, u)} for l, u in zip(lower_bound, upper_bound)]

    myProblem = GPyOpt.methods.BayesianOptimization(f_obj_wrapped,
                                                    X=np.atleast_2d(p0),
                                                    domain=bounds,
                                                    acquisition_type='MPI',
                                                    verbosity=True,
                                                    maximize=False,
                                                    num_cores=8
                                                    )

    myProblem.run_optimization(maxiter, verbosity=True)

    myProblem.save_evaluations("reports/evals.csv")
    myProblem.save_report("reports/report.csv")



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
