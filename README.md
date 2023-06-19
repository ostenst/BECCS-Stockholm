# The BECCS-Stockholm repository
This repo hosts an exploratory model of Stockholm Exergi AB's deployment of bioenergy cabon capture and storage (BECCS) in Stockholm, Sweden. The model.py file contains the investment decision model, and the view.py file contains different methods for analyzing the results. The controller.py file specifies and performs the complete analysis, when run in the terminal. The repo relies extensively on [Rhodium](https://github.com/Project-Platypus/Rhodium) [1] to perform the analysis.

# Installing and running the model
First make a clone of the BECCS-Stockholm repository. Then install Rhodium in your Python environment (we use Anaconda, Python v.3.10.9).

The latest Rhodium code has not yet been made available on PyPi, but can be installed using:

    pip install git+https://github.com/Project-Platypus/Rhodium@master#egg=Rhodium

Once it is made available on PyPi, install using:

    pip install rhodium

(Not recommended) it is also possible to install a custom version of Rhodium from @ostenst's fork:

    pip install git+https://github.com/ostenst/Rhodium@master#egg=Rhodium

To run the model, navigate to your BECCS-Stockholm directory and run controller.py in the terminal. If the model does not run, check other dependencies (e.g. numpy, numpy_financial, random, csv, openpyxl, matplotlib, Image). In the controller.py file, sample size can be lowered (e.g. to 10 000) for fast model evaluations.

[1] Hadjimichael A, et al. 2020 Rhodium: Python Library for Many-Objective Robust Decision Making and Exploratory Modeling. Journal of Open Research Software, 8: 12. DOI: https://doi.org/10.5334/jors.293
