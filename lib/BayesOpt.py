import moments
import numpy as np
import GPyOpt

BayesInference = moments.Inference


def optimize(p0, data, model_func, lower_bound=None, upper_bound=None,
             verbose=0, flush_delay=0.5, epsilon=1e-3,
             gtol=1e-5, multinom=True, maxiter=1000, full_output=False,
             func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
             output_file=None, output_dir=None):

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

    if fixed_params is None:
        bounds = [{'domain': (l, u)} for l, u in zip(lower_bound, upper_bound)]
    else:
        bounds = [{'domain': (l, u)} for l, u, p in zip(lower_bound, upper_bound, fixed_params) if p is None]

    myProblem = GPyOpt.methods.BayesianOptimization(f_obj_wrapped,
                                                    X=np.atleast_2d(p0),
                                                    domain=bounds,
                                                    acquisition_type='MPI',
                                                    verbosity=True,
                                                    maximize=False,
                                                    num_cores=24
                                                    )

    myProblem.run_optimization(maxiter, verbosity=True)

    _out_dir = ("out" + output_dir + "/") if output_dir is not None else ""
    myProblem.plot_acquisition(_out_dir +  "plots/acquisition.png")
    myProblem.plot_convergence(_out_dir + "plots/convergence.png")
    myProblem.save_evaluations(_out_dir + "reports/evals.tsv")
    myProblem.save_report(_out_dir + "reports/report.txt")

    xopt = BayesInference._project_params_up(myProblem.x_opt, fixed_params)
    xopt = BayesInference._project_params_down(xopt, fixed_params)

    outputs = scipy.optimize.fmin_bfgs(BayesInference._object_func, xopt,
                                       epsilon=epsilon,
                                       args=args, gtol=gtol,
                                       full_output=True,
                                       disp=False,
                                       maxiter=maxiter,
                                       fixed_params=fixed_params)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = BayesInference._project_params_up(xopt, fixed_params)

    return xopt
