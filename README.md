## Overview

Here are some questions I encounter in my day job that pop up again and again and are subtler than they appear. I'm attempting to follow E. T. Jaynes's advice of relentlessly dissecting real problems until I understand them.

* [The questions](#The-questions)
* [Prerequisites](#Prerequisites)
* [Usage](#Usage)

### The questions

* **Are today's sales lower than yesterday's because of random fluctuation, or should I be worried?**
  * problem description: [day_to_day_sales.ipynb](day_to_day_sales.ipynb)
  * report utility: TODO
* **Is today's conversion rate lower than yesterday's because of random fluctuation, or should I be worried?**
    * problem description: TODO
    * report utility: [binomial_model_comparison_utility.ipynb](binomial_model_comparison_utility.ipynb)
* **We have a timeseries of sales events: is there a point in time at which the sales rate suddenly changes?**
  * problem description: TODO
  * report utility: [switchpoint_report.ipynb](switchpoint_report.ipynb)
* **What's causing the current sales slump? Bayesian Search Theory: not just for finding nuclear subs**
  * problem description: [bayesian_search_theory.ipynb](bayesian_search_theory.ipynb)
  * utility: TODO

### Prerequisites

* python >= 3.6
* whatever the requirements in [requirements.txt](requirements.txt) need to build on your platform
* `make` will make things easier but you can do without it

### Usage

If you don't want to use make you can just install the requirements manually in your current environment with:
* `pip install -r requirements.txt` to install the requirements
* `jupyter notebook` to launch the notebook server

Using make is more convenient and keeps everything in a local virtualenv in the `env` directory:

* `make env_ok` to install virtualenv with all requirements
* `make run_notebook` to launch the notebook server and run the examples
* `make test` to run typechecks and unit tests
* `make fmt` to run formatters on the python code
* `make clean` to start over
