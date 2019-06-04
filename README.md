# Bayesian optimization for demographic history inference

*Bioinformatics Institute, 2019*

## Project goals

The project goal was to replace a [BFGS optimization algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) used in the tool [moments](https://bitbucket.org/simongravel/moments) with the Gaussian process based global Bayesian optimization and to study the effects. 

## Methods

We used [GPyOpt](https://github.com/SheffieldML/GPyOpt/) library that, given function and bounding conditions optimizes said function, then we took the optimization routine in `moments` and replaced their algorithm with the calls to `GPyOpt` and some other of our changes to make them work together.

To determine next query point for bayesian optimization we used `Expected Improvement` acquisition function, but it may be a subject for future rethinking.

## Requirements

Python >= 2.5

(Tested on Ubuntu 19.04, Python 2.7.15)

## How to run

### Dependencies

*It's best to use isolated environments like virtualenv or conda.*

Install [moments](https://bitbucket.org/simongravel/moments) according to the instructions, then execute

```
pip install -r requirements.txt
```

to install dependencies.

### Running

Run `python2 2pop_6.py`, `python2 2pop_8.py` or `python2 3pop_13.py`.

There are no parameters to set in CLI. You can change the program by editing the source code.

#### Clarification

Scripts (`2pop_6.py`, `2pop_8.py`, `3pop_13.py`) used employ standard Out of Africa hypothesis of human origin. They differ only in number of populations used (2 and 3 respectively) and number of parameters that a given model uses (6, 8 or 13).

Scripts try to produce the optimal parameters for each model that describe the demographics history for given population group.



## Results

After the successful run of the one of the aforementioned scripts you'll see the results in `out` folder. They are in the form of csv table `log.txt`, `GPyOpt` reports and `GPyOpt` plots provided that your problem is 1- or 2-dimensional.

From the log file convergence plot may be constructed, for example

![Convergence plot for 2pop_6](./org/reports/final_pres/plots/2pop_6.best.log.png)
