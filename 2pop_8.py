from __future__ import print_function

import moments
from numpy.random import uniform
import lib.BayesOpt as BayesOpt
import pylab
import numpy as np

np.random.seed(0)
def generate_random_value(low_bound, upp_bound, identificator=None):
    """Generate random value for different parameters of models"""
    if identificator == 't' or identificator == 's':
        return np.random.uniform(low_bound, upp_bound)
    mode = 1.0
    if low_bound >= mode:
        sample = np.random.triangular(low_bound, low_bound, upp_bound)
    elif upp_bound <= mode:
        sample = np.random.triangular(low_bound, upp_bound, upp_bound)
    else:
        sample = np.random.triangular(low_bound, mode, upp_bound)
    return sample

def model_func(params, ns):
    '''
    nu1A: The ancestral population size after growth. (Its initial size is
      defined to be 1.)
    s2B: The fraction of split: (1-s2B) will form pop2
    (nu2B = nu1A * s2B: The bottleneck size for pop2)
    nu2F: The final size for pop2
    nu1F: The final size for pop1
    m12: The scaled migration rate from pop2 to pop1
    m21: The scaled migration rate from pop1 to pop2
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present
    '''
#       print params
    nu1A, s2B, nu2F, nu1F, m12, m21, Tp, T = params
    nu2B = nu1A * (1 - s2B)

    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    Npop = lambda t: [nu1F]
    fs.integrate(Npop=Npop, tf=Tp)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

    growth_funcs = [lambda t: nu2F, lambda t: nu2B * (nu2F / nu2B) ** (t / T)]
    list_growth_funcs = lambda t: [ f(t) for f in growth_funcs]
    m = np.array([[0, m12],[m21, 0]])
    fs.integrate(Npop=list_growth_funcs, tf=T, m=m, dt_fac=0.1)

    return fs

data = moments.Spectrum.from_file('lib/YRI_CEU.fs')
ns = [20, 20]

ids =         ['n',  's',  'n',  'n',     'm',    'm',    't',     't'] 
upper_bound = [100,  100,  100,  100,      10,     10,      3,      3 ]
lower_bound = [1e-2, 1e-2, 1e-2, 1e-2,  1e-2,  1e-2,   1e-2, 1e-2]


optimal_params = [1.8638829321580523, 0.9582544565961783, 1.8823306815120924, 1.6838523185401713, 1.0197021070711312, 0.8774542996156008, 0.3729612223111333, 0.1176777311575127]

initial_design_size = 1
p0 = []
for _ in range(initial_design_size):
    cur_p0 = []
    for i, (cur_lower_bound, cur_upper_bound) in enumerate(zip(lower_bound, upper_bound)):
        cur_p0.append(generate_random_value(cur_lower_bound, cur_upper_bound, ids[i]))
    p0.append(cur_p0)

popt = BayesOpt.optimize(p0, data, model_func,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         verbose=1,
                         maxiter=5,
#                         fixed_params=popt1,
                         output_dir='2pop_8',
#                         log_params=False,
                         log_params=True,
                         
                         exact_feval = True,
#                         normalize_Y = True)
                         normalize_Y = False)
print(popt)

model = model_func(popt, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

