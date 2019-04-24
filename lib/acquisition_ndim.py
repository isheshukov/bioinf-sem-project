import matplotlib.pyplot as plt
import numpy as np


# TODO: заставить работать. сейчас не работает


def plot(gp, model_parameters=None):
    eval_X, eval_Y = gp.get_evaluations()

    ###
    ### ACQUISITION PROJECTION PLOTS
    ###
    if model_parameters is None:
        model_parameters = ["nuAf", "nuB", "nuEu0", "nuEu", "nuAs0", "nuAs", "mAfB", "mAfEu", "mAfAs", "mEuAs", "TAf", "TB", "TEuAs"]

    tot = len(model_parameters)
    cols = 3
    rows = tot // cols
    rows += tot % cols
    position = range(1, tot + 1)

    if gp.model.model is None:
        from copy import deepcopy
        model_to_plot = deepcopy(gp.model)
        if gp.normalize_Y:
            Y = gp.normalize(gp.Y, gp.normalization_type)
        else:
            Y = gp.Y
        model_to_plot.updateModel(gp.X, Y, gp.X, Y)
    else:
        model_to_plot = gp.model

    _plot_acquisition(gp.acquisition.space.get_bounds(),
                     model_to_plot.model,
                     model_to_plot.model.X,
                     model_to_plot.model.Y,
                     gp.acquisition.acquisition_function,
                     gp.suggest_next_locations(), None)

def _plot_acquisition(bounds, model, Xdata, Ydata, acquisition_function, suggested_sample, filename=None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    input_dim = 1

    x_grid = np.array([np.linspace(b[0], b[1], num=1000) for b in bounds]).T
    print("x_grid shape:", x_grid.shape)
    acqu = acquisition_function(x_grid)
    print("acqu shape:", acqu.shape)
    acqu_normalized = (-acqu - np.min(-acqu)) / (np.max(-acqu - np.min(-acqu)))
    print("acqu_normalized shape:", acqu_normalized.shape)

    px = x_grid[:, 0].reshape(x_grid.shape[0], 1)

    print(px)

    m, v = model.predict(x_grid.T)
    # print(m - 1.96 * np.sqrt(v))
    print("m shape:", m.shape)
    print("v shape:", v.shape)
    print("px shape:", px.shape)
    print("bounds shape:", bounds[0])
    model.plot_density(bounds[0], visible_dims=[0], alpha=.5)

    # print("Xdata", Xdata)
    # x_grid = x_grid[0].reshape(len(x_grid[0]), 1)

    plt.plot(px, m, 'k-', lw=1, alpha=0.6)
    plt.plot(px, m - 1.96 * np.sqrt(v), 'k-', alpha=0.2)
    plt.plot(px, m + 1.96 * np.sqrt(v), 'k-', alpha=0.2)

    plt.plot(Xdata[:, 0], Ydata, 'r.', markersize=10)

    print("sug sample shape:", suggested_sample.shape)
    print("sug sample", suggested_sample)

    plt.axvline(x=np.atleast_2d(suggested_sample[0][0]), color='r')
    factor = np.max(m + 1.96 * np.sqrt(v)) - np.min(m - 1.96 * np.sqrt(v))

    plt.plot(px, 0.2 * factor * acqu_normalized - abs(np.min(m - 1.96 * np.sqrt(v))) - 0.25 * factor, 'r-', lw=2,
             label='Acquisition (arbitrary units)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(np.min(m - 1.96 * np.sqrt(v)) - 0.25 * factor, np.max(m + 1.96 * np.sqrt(v)) + 0.05 * factor)
    plt.axvline(x=np.atleast_2d(suggested_sample[0][0]), color='r')
    plt.legend(loc='upper left')

    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
