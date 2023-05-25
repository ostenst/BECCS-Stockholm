"""ROBUST DECISION MAKING (RDM) MODEL FOR BECCS

This program was developed to perform a Robust Decision Making (RDM) analysis of the investment
decision of deploying a bioenergy carbon capture and storage deployment (BECCS) facility. The
case studied is the planned BECCS facility of Stockholm Exergi (more info at www.beccs.se).
The program utilizes Rhodium, an open-source Python library for RDM and exploratory modeling:
https://github.com/Project-Platypus/Rhodium
To run the program, first follow the installation guide provided in the repository. Make sure
this file "BECCS_investment_paper_version.py" is in the root "Rhodium" folder.

Glossary
--------
NE       = negative emission, i.e. one tonne of atmospheric CO2 permanently removed and stored
CF       = cash flow
p        = price
dt       = "likely" change over time (high value => higher probability of the parameter increasing
                                       low value => higher probability of the parameter decreasing)
y        = year
AUCTION  = a percentage of specific BECCS costs covered by a reversed auctions policy
yQUOTA   = a year when quota obligations of NEs could be imposed
yEUint   = a year when NEs could be integrated in an EU scheme for carbon removal trading
yBIOban  = a year when the usage of biomass as renewable energy could be severely restricted
yCLAIM   = a year when the claiming and selling of NEs to the voluntary carbon market is allowed
SOW      = state of the world, i.e. one specific combination of conditions (parameter values)
-Invest- = when emphasized, Invest refers to the decision to invest in the BECCS facility
-Wait-   = when emphasized, Wait refers to the decision to not invest in the BECCS facility

"""
__author__ = "Oscar Stenström"
__date__ = "2023-02-16"

import math
import numpy as np
import numpy_financial as npf
from scipy.optimize import brentq as root
from rhodium import Model, Parameter, Response, RealLever, UniformUncertainty
import random

from dataclasses import dataclass, field

## CONSTRUCT THE CALCULATION MODEL
# The calculations below aim to quantify the annual prices of electricity, heat and NEs (i.e. the
# revenues), as well as the different costs of the Invest and Wait strategies. This is done for 27
# time slots, as the model runs from 2024 (year 0) to 2050 (year 26). These values are then used to
# calculate the Net Present Value (NPV) of both strategies.

# One can set investment_decision = 1 if Invest is of interest, or investment_decision = 0 if Wait
# is of interest.

# One can set pelectricity_dt = [-1,0.4] depending on if trends of lower or higher electricity
# prices are explored, or pheat_dt = [-0.5,0.7] depending on if trends of lower or higher heat
# prices are explored.

## DEFINE HELPING FUNCTIONS:
def calculate_regret(
    NPV_invest: float, NPV_wait: float, investment_decision: int
) -> float:
    """Determine regret based on investment decision

    Arguments
    ---------
    NPV_invest: float
        Net present value if investing
    NPV_wait: float
        Net present value of waiting
    investment_decision: int
        1 means Invest and 0 means Wait.

    Returns
    -------
    float
        The regret is the difference between the decision and the max of the two strategies
    """
    if investment_decision == 0:
        Regret = max(NPV_invest, NPV_wait) - NPV_wait

    if investment_decision == 1:  # NOTE: this is the decision in focus for the article.
        Regret = max(NPV_invest, NPV_wait) - NPV_invest

    return Regret

def find_pETS(pETS_2050, pETS_dt):
    """Creates a price trajectory for EU ETS allowances

    Arguments
    ---------
    pETS_2050 : float
    pETS_dt : float

    Returns
    -----
    list

    Notes
    -----
    The price trajectory is a simple line between the ETS price in the first modelling year (about EUR 100 in 2023):
    https://tradingeconomics.com/commodity/carbon
    and the ETS price in the last modelling year, pETS_2050. The trajectory is perturbed by the price volatility pETS_dt.
    """
    pETS_2024 = 100
    pETS_vec = [pETS_2024]
    for t in range(1,27):
        pETS_new = (pETS_2050-pETS_2024)/(2050-2024)*t + pETS_2024
        pchange = random.uniform(-pETS_dt,pETS_dt)
        pETS_new = pETS_new * (1 + pchange)
        pETS_vec.append(pETS_new)
    return pETS_vec

def find_sell_prices(pmean, pvolatility, pfloor, ySHOCK):
    """Determines selling prices of electricity, heat and NEs

    Arguments
    ---------
    pmean : float
    pvolatility : float
    pfloor : float
    
    Returns
    -----
    list

    Notes
    -----
    The price for each year is the mean price, perturbed by the price volatility, constrained by a price floor.
    """
    pvec = []
    for t in range(0,27):
        pchange = random.uniform(-pvolatility,pvolatility)
        if pmean + pchange < pfloor:
            pnew = pfloor
        else:
            pnew = pmean * (1 + pchange)

        # If a year of a price shock is reached, new prices are temporarily heightened by ~80 %. This assumption is in-line with the historic electricity prices of the Stockholm area in 2022:
        # https://www.vattenfall.se/elavtal/elpriser/rorligt-elpris/prishistorik/
        if 2024+t == round(ySHOCK):
            pnew = pnew*1.8

        pvec.append(pnew)
    return pvec

@dataclass(slots=True)
class BeccsPlant:
    """Represents a biomass CCS plant

    POWER PLANT OPERATING CONDITIONS: Operating conditions are not modelled as uncertainties,
    as they are routinely managed by the power plant operators.
    Refer to full article for data sources.

    Arguments
    ---------
    Qbiomass_input : float
    Wpower_output_wait : float
    Qheat_output_wait : float
    Wpower_output_invest : float
    Qheat_output_invest : float
    Operating_hours : float
    CO2capture_rate : float

    Attributes
    ----------
    CO2captured: float
    OPEX_power_plant: float
    """

    Qbiomass_input: float = 400  # [MW]
    Wpower_output_wait: float = 118  # [MW]
    Qheat_output_wait: float = 330  # [MW]
    Wpower_output_invest: float = 40  # [MW]
    Qheat_output_invest: float = 424  # [MW]

    Operating_hours: float = (
        8760 * 0.7
    )  # [h] rule of thumb of ~70 % operating hours/year.
    # CO2capture_rate: float = 0.3  # [tCO2/MWh_biomass]
    CO2capture_rate: float = 140  # [tCO2/h]

    CO2captured: float = field(init=False)  # [tCO2/year]
    OPEX_power_plant: float = field(init=False)  # [EUR/year]

    # [tCO2/year] NOTE: SHOULD BE ABOUT 33% HIGHER
    # CO2captured = CO2capture_rate * Qbiomass_input * Operating_hours 
    CO2captured = CO2capture_rate * Operating_hours
    
    # [EUR/year], see full article for these operational costs.
    OPEX_power_plant = 29000 * Qbiomass_input + 0.5 * Operating_hours * Qbiomass_input


def calculate_cash_flow(
    year_index: int,
    plant: BeccsPlant,
    electricity_price: list,
    heat_price: list,
    biomass_price: float,
    wait: bool = True,
) -> float:
    """Calculate the cash flow

    Arguments
    ---------
    year_index: int
    plant: BeccsPlant
    electricity_price: list
    heat_price: list
    biomass_price: float
    wait: bool, default=True
        Wait if True, Invest if False

    Returns
    -------
    float
    """
    if wait == True:
        power_output = plant.Wpower_output_wait
        heat_output = plant.Qheat_output_wait
    elif wait == False:
        power_output = plant.Wpower_output_invest
        heat_output = plant.Qheat_output_invest

    biomass_input = plant.Qbiomass_input
    operating_hours = plant.Operating_hours
    opex = plant.OPEX_power_plant

    cash_flow = (
        power_output * electricity_price[year_index]
        + heat_output * heat_price[year_index]
        - biomass_input * biomass_price
    ) * operating_hours - opex
    return cash_flow


def BECCS_investment(
    investment_decision,
    # Set some nominal values as function inputs, as desired.
    pelectricity_mean=50,  # [EUR/MWh]
    pheat_mean=50,  # [EUR/MWh]
    pNE_mean=30,  # [EUR/tCO2]
    pETS_2050=80,  # [EUR/tCO2]
    pbiomass=25,  # [EUR/MWh]
    pelectricity_dt=0.4,  # -1 to 0.4,   sets likelihood of price increase/reduction #NOTE: CHANGE THIS
    pheat_dt=-0.5,  # -0.5 to 0.7, sets likelihood of price increase/reduction
    pNE_dt=0.3,  # -1 to +1,    sets likelihood of price increase/reduction
    pETS_dt=0,  # -1 to +1,    sets exponential increase in ETS prices
    Discount_rate=0.06,  # [-]
    Learning_rate=0.01,  # [-]
    Availability_factor=0.7, # [-]
    CAPEX=200 * 10**6,  # [EUR]
    OPEX_fixed=20 * 10**6,  # [EUR/year]
    OPEX_variable=44,  # [EUR/tCO2]
    Cost_transportation=22,  # [EUR/tCO2]
    Cost_storage=14.5,  # [EUR/tCO2]
    AUCTION=0.5,  # [-]
    yQUOTA=2035,
    yEUint=2040,
    yBIOban=2051,
    yCLAIM=2026,
    ySHOCK=2051,
):
    """Calculate regret and other metrics for two strategies "wait" or "invest" for a SOW

    Notes
    -----
    First generates price trajectories
    Then computes NPV for wait and invest strategies
    Finally, computes metrics
    """

    plant = BeccsPlant(Operating_hours=Availability_factor*8760)

    # CALCULATE ENERGY/CO2 PRICES FOR THIS SOW:
    # Helping functions are used to construct pseudo-random energy and CO2 price projections. The
    # probabilities of price increases/decreases are influenced by the _dt parameters.

    pelectricity = find_sell_prices(
        pelectricity_mean,
        pelectricity_dt,
        pfloor=5,
        ySHOCK=ySHOCK,
    )
    pheat = find_sell_prices(
        pheat_mean,
        pheat_dt,
        pfloor=48,
        ySHOCK=2051,
    )
    pNE = find_sell_prices(
        pNE_mean,
        pNE_dt,
        pfloor=3,
        ySHOCK=2051,
    )
    pETS = find_pETS(
        pETS_2050, 
        pETS_dt)
    # It is now possible to calculate NPV values!

    # CALCULATE CASH FLOWS BASED ON BOTH INVESTMENT DECISIONS:
    # (1) calculate NPV for not investing, i.e. Waiting:
    CFvec_wait = []

    NPV_wait = 0
    for t in range(0, 27):
        CF = calculate_cash_flow(t, plant, pelectricity, pheat, pbiomass, wait=True)
        if 2024 + t >= yBIOban:
            CF -= (
                plant.CO2captured * pETS[t]
            )  # yBIOban represents a severe restriction of biomass usage, forcing the utility to pay for emission allowances for CO2 not captured.
        NPV_wait += CF / ((1 + Discount_rate) ** t)
        CFvec_wait.append(CF)
    # Now NPV is known for the Wait strategy!

    # (2) calculate NPV for the Invest strategy:
    NPV_invest = 0
    CFvec = []
    # In this vector we save pNE_max, which is the maximum CO2 price offered from any of the different support policy models.
    pNE_supported = []
    # First two years we pay CAPEX, and we do not have revenues from NEs.
    for t in range(0, 2):
        CF = (
            calculate_cash_flow(t, plant, pelectricity, pheat, pbiomass, wait=True)
            - CAPEX / 2
        )
        CFvec.append(CF)

        # The first two years, the maximum CO2 price is just the VCM price pNE(t).
        pNE_supported.append(pNE[t])

    for t in range(2, 27):
        CFenergy = calculate_cash_flow(
            t, plant, pelectricity, pheat, pbiomass, wait=False
        )
        Cost_specific = (
            (OPEX_variable + Cost_transportation + Cost_storage)
            + OPEX_fixed / plant.CO2captured
        ) * (1 - Learning_rate * (t - 2))

        # Now, what is the maximum NE price we can sell to in this SOW?
        # Answer: the highest of the VCM price, and prices achieved from uncertain policy support:
        pNE_max = pNE[t]
        # If quota obligations (yQUOTA) exist, NE price is assumed to be _at least_ equal to the specific cost.
        if 2024 + t >= yQUOTA:
            if Cost_specific > pNE_max:
                pNE_max = Cost_specific
        # If ETS integration (yEUint) exist, we sell to a price equivalent to ETS levels, if that price is higher.
        if 2024 + t >= yEUint:
            if pETS[t] > pNE_max:
                pNE_max = pETS[t]
        # Reversed auctions (AUCTION) reduces the _specific_ costs, until 2040.
        if 2024 + t <= 2040:
            Cost_specific = Cost_specific * (1 - AUCTION)

        # It is now possible to calculate cash flows from the maximum pNE offered, CFCO2.
        # However: we can not sell NEs (i.e. price is set to zero) if EU severely restricts biomass usage (yBIOban), or if we can't claim NEs (yCLAIM):
        if (2024 + t < yBIOban) and (2024 + t > yCLAIM):
            CFCO2 = pNE_max * plant.CO2captured - (Cost_specific * plant.CO2captured)
        else:
            pNE_max = 0
            CFCO2 = pNE_max * plant.CO2captured - (Cost_specific * plant.CO2captured)

        # Now when cash flows from energy and CO2 is known, NPV of Investing can be calculated:
        CF = CFenergy + CFCO2
        NPV_invest += CF / ((1 + Discount_rate) ** t)
        CFvec.append(CF)
        pNE_supported.append(pNE_max)

    # Now NPV is known for the Invest strategy as well! Regret can be calculated.
    Regret = calculate_regret(NPV_invest, NPV_wait, investment_decision)

    ## CALCULATE OTHER INTERESTING PARAMETERS:
    # pelectricity_mean = np.mean(pelectricity)
    # pheat_mean = np.mean(pheat)
    pNE_supported = np.mean(pNE_supported) # THIS IS OUTDATED
    Cost_specific = (
            (OPEX_variable + Cost_transportation + Cost_storage)
            + OPEX_fixed / plant.CO2captured
        ) 

    # Now the calculation model is done!
    return (NPV_invest, NPV_wait, Regret, pbiomass, pelectricity, pheat, pNE, pETS, CFvec, Cost_specific, pNE_supported)


def return_model() -> Model:
    ## DEFINE RHODIUM MODEL
    model = Model(BECCS_investment)
    model.parameters = [
        Parameter("investment_decision"),
        Parameter("pNE_mean"),
        Parameter("pNE_dt"),
        Parameter("pelectricity_mean"),
        Parameter("pelectricity_dt"),
        Parameter("pheat_mean"),
        Parameter("pheat_dt"),
        Parameter("pbiomass"),

        Parameter("pETS_2050"),
        Parameter("pETS_dt"),
        Parameter("Discount_rate"),
        Parameter("CAPEX"),
        Parameter("OPEX_fixed"),
        Parameter("OPEX_variable"),
        Parameter("Cost_transportation"),
        Parameter("Cost_storage"),
        Parameter("Learning_rate"),
        Parameter("Availability_factor"),
        
        Parameter("AUCTION"),
        Parameter("yQUOTA"),
        Parameter("yEUint"),
        Parameter("yBIOban"),
        Parameter("yCLAIM"),
        Parameter("ySHOCK"),
    ]

    model.responses = [
        Response("NPV_invest", Response.MAXIMIZE),
        Response("NPV_wait", Response.MAXIMIZE),
        Response("Regret", Response.MINIMIZE),
        Response("pbiomass", Response.INFO),
        Response("pelectricity", Response.INFO),
        Response("pheat", Response.INFO),
        Response("pNE", Response.INFO),
        Response("pETS", Response.INFO),
        Response("CFvec", Response.INFO),
        Response("Cost_specific", Response.INFO),
        Response("pNE_supported", Response.INFO),
    ]

    model.levers = [RealLever("investment_decision", 0, 1, length=2)]

    # For uncertainties, some are expanded ranges around values found in the literature, and some are assumed. Refer to full article.
    model.uncertainties = [
        UniformUncertainty("pNE_mean", 30, 300), #20-400 or 100-500... oooor: 100-300(DACCS upper cost). If 30 lower end, justify by referring to ETS=100.
        UniformUncertainty("pNE_dt", 0.01, 0.50),
        UniformUncertainty("pbiomass", 15, 35), #15-35
        UniformUncertainty("pelectricity_mean",20,160), #KÅRE USED THIS
        UniformUncertainty("pelectricity_dt",0.01,0.50),
        UniformUncertainty("pheat_mean",50, 150), #50 (from Exergi) or 40 (from Kåre)?
        UniformUncertainty("pheat_dt",0.01,0.50),

        UniformUncertainty("pETS_2050", 125, 375), #100-900, but Implement Consulting suggest others... IEA suggest 250? https://iea.blob.core.windows.net/assets/2db1f4ab-85c0-4dd0-9a57-32e542556a49/GlobalEnergyandClimateModelDocumentation2022.pdf 
        UniformUncertainty("pETS_dt", 0.01, 0.50),
        UniformUncertainty("Discount_rate", 0.04, 0.10), 
        UniformUncertainty("CAPEX", 100 * 10**6, 300 * 10**6),
        UniformUncertainty("OPEX_fixed", 10 * 10**6, 30 * 10**6), #18-22 or 10-30 (mean 20)
        UniformUncertainty("OPEX_variable", 18.5, 55.5),    #33-55 or 22-66 (mean 44)... but this contains elec penalty? NEW: Use mean 37, but motivate +- heat/elec prices. Beiron: 15-30.
        UniformUncertainty("Cost_transportation", 17, 27),
        UniformUncertainty("Cost_storage", 6, 23),
        UniformUncertainty("Learning_rate", 0.0075, 0.0125),
        UniformUncertainty("Availability_factor", 0.65, 0.75),

        UniformUncertainty("AUCTION", 0, 1),
        UniformUncertainty("yQUOTA", 2030, 2050),
        UniformUncertainty("yEUint", 2035, 2050),
        UniformUncertainty("yBIOban", 2030, 2050),
        UniformUncertainty("yCLAIM", 2024, 2050),
        UniformUncertainty("ySHOCK", 2030, 2050),
    ]
    return model
