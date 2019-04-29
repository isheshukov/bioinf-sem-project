import moments
import numpy as np
import GPyOpt
import os
from pathlib2 import Path

BayesInference = moments.Inference


def optimize(p0, data, model_func, lower_bound=None, upper_bound=None,
             verbose=0, flush_delay=0.5, epsilon=1e-3,
             gtol=1e-5, multinom=True, maxiter=1000, full_output=False,
             func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
             output_file=None, output_dir=None, plot_every=None, use_bfgs=False):

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

    # Fixing parameters. 
    p0 = BayesInference._project_params_down(p0, fixed_params)

    if fixed_params is None:
        bounds = [{'domain': (l, u)} for l, u in zip(lower_bound, upper_bound)]
    else:
        bounds = [{'domain': (l, u)} for l, u, p in zip(lower_bound, upper_bound, fixed_params) if p is None]

    out_dir_path = Path('out') / output_dir
    plots_dir = out_dir_path / "plots"
    acquisition_dir = plots_dir / "acquisitions"
    reports_dir = out_dir_path / "reports"

    plots_dir.mkdir(parents=True, exist_ok=True)
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


    # Running optimization in a loop. Plot acquisition graph every
    # `plot_every` iteration

    opt = GPyOpt.methods.BayesianOptimization(f_obj_wrapped,
                                              X=np.atleast_2d(p0),
                                              domain=bounds,
                                              acquisition_type='MPI',
                                              verbosity=True,
                                              maximize=False,
                                              num_cores=8
                                              )
    for current_iter in range(maxiter):
        opt.run_optimization(max_iter=1)
        if (plot_every is not None) and (current_iter % plot_every == 0):
            opt.plot_acquisition(
                (acquisition_dir / 'acquisition_{0:04d}'.format(current_iter)).as_posix()
                )

    opt.plot_acquisition((plots_dir / "acquisition.png").as_posix())
    opt.plot_convergence((plots_dir / "convergence.png").as_posix())
    opt.save_evaluations((reports_dir / "evals.tsv").as_posix())
    opt.save_report((reports_dir / "report.txt").as_posix())

    xopt = BayesInference._project_params_up(opt.x_opt, fixed_params)
   
    if use_bfgs == True:
        xopt = moments.Inference.optimize(xopt, data, model_func,
                                        lower_bound=lower_bound,
                                        upper_bound=upper_bound,
                                        fixed_params=fixed_params,
                                        maxiter=maxiter,
                                        verbose=1)
    return xopt
