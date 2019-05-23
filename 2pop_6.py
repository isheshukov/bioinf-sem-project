from __future__ import print_function

import moments
from numpy.random import uniform
import lib.BayesOpt as BayesOpt
import pylab
import numpy as np

np.random.seed(0)

from lib.demographic_models import prior_onegrow_mig

# p_ids is list of identifiers for parameters:
#     t - time
#     n - size of population
#     m - migration
# or it can be None

# Actually we use triangular distribution for all parameters except time.
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

data = moments.Spectrum.from_file('lib/YRI_CEU.fs')
ns = data.sample_sizes

# Parameters are: (nu1f, nu2B, nu2F, m, Tp, T)

ids =         ['n',  'n',  'n',  'm',  't',  't'] 
upper_bound = [100,  100,  100,  10,    3,    3 ]
lower_bound = [1e-2, 1e-2, 1e-2,  1e-2,    1e-4,    1e-4 ]

func = prior_onegrow_mig
#################p0 = [uniform(l, u) for l, u in zip(lower_bound, upper_bound)]
initial_design_size = 1
p0 = []
for _ in range(initial_design_size):
    cur_p0 = []
    for i, (cur_lower_bound, cur_upper_bound) in enumerate(zip(lower_bound, upper_bound)):
        cur_p0.append(generate_random_value(cur_lower_bound, cur_upper_bound, ids[i]))
    p0.append(cur_p0)

# We'll work with them through the rest of this script.
popt = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]

# Fixing first two params
optimal_params = np.array([1.881, 0.0710, 1.845, 0.911, 0.355, 0.111])
print('Optimal params')
print(', '.join('%0.3f' % one_arg for one_arg in optimal_params))
print('Log optimal params')
print(', '.join('%0.3f' % one_arg for one_arg in np.log(optimal_params)))
#print(np.log(optimal_params))
print('')

print('========================== Beginning optimization =================================')
print('')


###
#import sys
#import inspect
#
#class PrintSnooper:
#    def __init__(self, stdout):
#        self.stdout = stdout
#    def caller(self):
#        return inspect.stack()[2][3]
#    def write(self, s):
#        self.stdout.write("printed by %s: " % self.caller())
#        self.stdout.write(s)
#        self.stdout.write("\n")
#
## redirect stdout to a helper class.
#sys.stdout = PrintSnooper(sys.stdout)
###

#popt = BayesOpt.optimize(p0, data, func,
#                         lower_bound=lower_bound,
#                         upper_bound=upper_bound,
#                         verbose=1,
#                         maxiter=5,
##                         fixed_params=popt1,
#                         output_dir='2pop_6',
##                         log_params=False,
#                         log_params=True,
#                         
#                         exact_feval = True,
##                         normalize_Y = True)
#                         normalize_Y = False)
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

