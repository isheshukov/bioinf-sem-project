import pandas as pd
import matplotlib.pyplot as plt

gadma_files = ["13_params_GADMA_data.csv", "6_params_GADMA_data.csv"]
files = ["3pop_13.best.log", "2pop_6.best.log"]


for g, f in zip(gadma_files, files):
    gadma = pd.read_csv("data/" + g)
    gpy = pd.read_csv("data/" + f)
    plt.clf()
    plt.plot(gadma.values[:2000, 1], label='GADMA')
    plt.plot(gpy.values[:2000,0], label='GPyOpt')
    plt.ylim((0,35000))
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.legend()
    plt.savefig(f + ".png", dpi=300)
    plt.clf()
