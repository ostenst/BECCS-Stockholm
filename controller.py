"""ROBUST DECISION MAKING (RDM) MODEL FOR BECCS

This program was developed to perform a Robust Decision Making (RDM) analysis of the investment
decision of deploying a bioenergy carbon capture and storage deployment (BECCS) facility. The
case studied is the planned BECCS facility of Stockholm Exergi (more info at www.beccs.se).
The program utilizes Rhodium, an open-source Python library for RDM and exploratory modeling:
https://github.com/Project-Platypus/Rhodium
To run the program, first follow the installation guide provided in the repository. Make sure
this file "BECCS_investment_paper_version.py" is in the root "Rhodium" folder.
"""
__author__ = "Oscar Stenström"
__date__ = "2023-02-16"

from scipy.optimize import brentq as root
from rhodium import Model, sample_lhs, update, evaluate, sa, DataSet
import openpyxl
import random

from model import return_model
from view import (
    plot_results,
    scenario_discovery,
    save_model_results,
    save_scenario_discovery,
    plot_scenario_of_interest,
    save_sensitivity_analysis,
    plot_sensitivity_analysis_results,
    plot_critical_uncertainties,
    robustness_analysis,
)

# investment_decision = 1 means Invest and investment_decision = 0 means Wait.
POLICY = {"investment_decision": 1}


def evaluate_model(model: Model) -> DataSet:
    """Evaluates the model with a latin hypercube sample

    Returns
    -------
    list
        A Rhodium dataset (list of dict)
    """
    random.seed(7)

    # 10 000 SOWs are evaluated, and in the full article this is repeated for
    # four energy price trends (i.e. by changing `pelectricity_dt` and `pheat_dt`).
    SOWs = sample_lhs(model, 10000)
    inputs = update(SOWs, POLICY)
    model_results = evaluate(model, inputs)
    return model_results


def conduct_sensitivity_analysis(model: Model, policy):
    print("-------------BEGIN SENSITIVITY ANALYSIS NOW-------------")
    # The sensitivity analysis indicates what uncertainties drive Regret. Using Sobols method, this is measured in 1st, 2nd and total order sensitivity indices.
    # The article uses nsamples = 500 000, but for fast model evaluations nsamples = 10 000 can be used.
    sobol_result = sa(model, "Regret", policy=policy, method="sobol", nsamples=10000)
    return sobol_result


def main():
    """Conduct analysis"""

    # Create Excel workbook to house results
    workbook = openpyxl.Workbook()

    model = return_model()
    model_results = evaluate_model(model)
    save_model_results(workbook, model_results)
    plot_results(model, model_results)
    robustness_analysis(model_results)
    node_list = scenario_discovery(model, model_results) #TODO: MAKE SURE SCENARIOS ARE DISCOVERED AND SAVED
    #save_scenario_discovery(node_list, workbook)
    plot_scenario_of_interest(model, model_results)
    sa_result = conduct_sensitivity_analysis(model, POLICY)
    # You can comment out the CART node prints and the Sobol sensitivity analysis prints,
    # as these anyway are saved to the RDM_processed_results file.
    # Doing this makes it easier to see the robustness results, which are printed in the terminal."
    save_sensitivity_analysis(model, sa_result, workbook)
    plot_sensitivity_analysis_results(sa_result)
    plot_critical_uncertainties(model, model_results)
    print("\n-------------END OF MODEL-------------")


if __name__ == "__main__":

    main()
