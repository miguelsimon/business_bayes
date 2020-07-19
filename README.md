## Overview

Here are some problems I encounter in my day job that pop up again and again and are subtler than they appear. I'm attempting to follow E. T. Jaynes's advice of relentlessly dissecting real problems until I understand them.

The report utilities are ipython notebooks, prepared so you can just plug in the numbers to get quick answers.

* [Index](#Index)
  * [Changes in sales over time](#Changes-in-sales-over-time)
  * [Conversion rate changes](#Conversion-rate-changes)
  * [Finding change points in sales timeseries](#Finding-change-points-in-sales-timeseries)
  * [Finding bugs with Bayesian Search Theory](#Finding-bugs-with-Bayesian-Search-Theory)
* [Prerequisites](#Prerequisites)
* [Usage](#Usage)

### Index

#### Changes in sales over time

  Are today's sales lower than yesterday's because of random fluctuation, or is it likely there's an underlying cause for the change?

  * problem description (TODO: fix this): [day_to_day_sales.ipynb](day_to_day_sales.ipynb)
  * model comparison math: [poisson_model_comparison_derivation.ipynb](poisson_model_comparison_derivation.ipynb)
  * report utility: [poisson_model_comparison_utility.ipynb](poisson_model_comparison_utility.ipynb)

#### Conversion rate changes

  Are the changes in conversion rate I'm seeing due to random fluctuations, or is it likely there's an underlying cause?

  * problem description: TODO
  * report utility: [binomial_model_comparison_utility.ipynb](binomial_model_comparison_utility.ipynb)

#### Finding change points in sales timeseries

  Given a timeseries of sales over time, can we find a point where the sales rate changes significantly?

  Current implementation finds 1 possible change point, need to get around to implementing reversible jump markov chain monte carlo for the general case as described in [Reversible Jump Markov Chain Monte Carlo Computation and Bayesian Model Determination](https://doi.org/10.1093%2Fbiomet%2F82.4.711) by Green.

  * problem description: TODO
  * report utility: [switchpoint_report.ipynb](switchpoint_report.ipynb)

#### Finding bugs with Bayesian Search Theory

  Bayesian search theory is not just for finding nuclear subs.

  * problem description: [bayesian_search_theory.ipynb](bayesian_search_theory.ipynb)
  * utility: TODO

### Prerequisites

* python >= 3.6
* whatever the requirements in [requirements.txt](requirements.txt) need to build on your platform
* `make` will make things easier but you can do without it

### Usage

If you don't want to use make you can just install the requirements manually in your current environment with:
* `pip3 install -r requirements.txt` to install the requirements
* `jupyter notebook` to launch the notebook server

Using make is more convenient and keeps everything in a local virtualenv in the `env` directory:

* `make env_ok` to install virtualenv with all requirements
* `make run_notebook` to launch the notebook server and run the examples
* `make test` to run typechecks and unit tests
* `make fmt` to run formatters on the python code
* `make clean` to start over
