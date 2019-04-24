from GPyOpt.methods import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import pysnooper


# --- Define your problem
from GPyOpt.util.general import normalize

@pysnooper.snoop()
def plot_acquisition(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''


    x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
    print("x_grid shape:", x_grid.shape)
    x_grid = x_grid.reshape(len(x_grid),1)
    print("x_grid shape:", x_grid.shape)
    acqu = acquisition_function(x_grid)
    print("acqu shape:", acqu.shape)
    acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
    print("acqu_normalized shape:", acqu_normalized.shape)
    m, v = model.predict(x_grid)
    print("m shape:", m.shape)
    print("v shape:", v.shape)

    model.plot_density(bounds[0], alpha=.5)

    print("bounds shape:", bounds)
    print("sug sample", suggested_sample)
    plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
    plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
    plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

    plt.plot(Xdata, Ydata, 'r.', markersize=10)
    plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
    factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

    plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
    plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
    plt.legend(loc='upper left')


    if filename!=None:
        plt.savefig(filename)
    else:
        plt.show()

def f(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=15)

if myBopt.model.model is None:
    from copy import deepcopy

    model_to_plot = deepcopy(myBopt.model)
    if myBopt.normalize_Y:
        Y = normalize(myBopt.Y, myBopt.normalization_type)
    else:
        Y = myBopt.Y
    model_to_plot.updateModel(myBopt.X, Y, myBopt.X, Y)
else:
    model_to_plot = myBopt.model

plot_acquisition(myBopt.acquisition.space.get_bounds(),
                 model_to_plot.model.X.shape[1],
                 model_to_plot.model,
                 model_to_plot.model.X,
                 model_to_plot.model.Y,
                 myBopt.acquisition.acquisition_function,
                 myBopt.suggest_next_locations(),
                 None)
#myBopt.plot_acquisition()