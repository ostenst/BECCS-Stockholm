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
# calculate the Net Present Value (NPV) of both strategies, and Internal Rate of Return (IRR) (only
# real solutions) of the Invest strategy.

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

def find_sell_prices(pmean, pvolatility, pfloor):
    # TODO: change pvolatility to be %-based instead. Makes it more flexible.
    pvec = []
    for t in range(0,27):
        # Hur mycket ska vi ändra pmean? Kör taket 160+40. Men floor?
        pchange = random.uniform(-pvolatility,pvolatility)
        if pmean + pchange < pfloor:
            pnew = pfloor
        else:
            pnew = pmean + pchange
        pvec.append(pnew)
    return pvec


def find_pNE(pNE, pNE_dt, pfloor, proof, pmaximum_increase, pmaximum_decrease):
    """Create a randomised price trajectory for emissions

    Arguments
    ---------
    pNE : float
    pNE_dt : float
        Randomisation factor in range [-1, +1].
        -1 means low values preferred.
        +1 high values preferred
    pfloor : float
        Price floor
    proof : float
        Price roof
    pmaximum_increase : float
    pmaximum_decrease : float
    """
    # Translates pNE_dt to an alpha/beta value:
    # These rows determine the "strength" of the _dt uncertainty. Can be experimented with to explore other price paths.
    alpha = (1.4 - 0.6) / 2 * (pNE_dt + 1) + 0.6
    beta = (0.6 - 1.4) / 2 * (pNE_dt + 1) + 1.4

    pNE_vec = []
    for t in range(0, 27):
        pNE_vec.append(pNE)
        # This is how much the price change will be scaled, from this year to the next...
        variance = random.betavariate(alpha * 1.4, beta * 1)

        # ... unless we hit a price roof/floor! Then adjust price accordingly.
        if pNE + pmaximum_increase > proof:
            pvariance = (
                pmaximum_decrease + ((proof - pNE) - pmaximum_decrease) * variance
            )
        elif pNE + pmaximum_decrease < pfloor:
            pvariance = (
                -(pNE - pfloor) + (pmaximum_increase - (-(pNE - pfloor))) * variance
            )
        else:
            pvariance = (
                pmaximum_decrease + (pmaximum_increase - pmaximum_decrease) * variance
            )
        pNE += pvariance

    return pNE_vec


# TODO: Combine or inherit from pNE as the approach is identical
def find_penergy(
    pelectricity, pelectricity_dt, pfloor, proof, pmaximum_increase, pmaximum_decrease
):
    """Create a randomised price trajectory for electricity

    Arguments
    ---------
    pelectricity : float
    pelectricity_dt : float
        Randomisation factor in range [-1, +1].
        -1 means low values preferred.
        +1 high values preferred
    pfloor : float
        Price floor
    proof : float
        Price roof
    pmaximum_increase : float
    pmaximum_decrease : float
    """
    # Works just like find_pNE()
    x = pelectricity_dt
    alpha = (1.7 - 0.3) / 2 * (x + 1) + 0.3
    beta = (0.3 - 1.7) / 2 * (x + 1) + 1.7

    pelectricity_vec = []
    for t in range(0, 27):
        pelectricity_vec.append(pelectricity)
        variance = random.betavariate(
            alpha * 1, beta * 1
        )  # *1.5 adds less random distribution, if desired.

        if pelectricity + pmaximum_increase > proof:
            pvariance = (
                pmaximum_decrease
                + ((proof - pelectricity) - pmaximum_decrease) * variance
            )
        elif pelectricity + pmaximum_decrease < pfloor:
            pvariance = (
                -(pelectricity - pfloor)
                + (pmaximum_increase - (-(pelectricity - pfloor))) * variance
            )
        else:
            pvariance = (
                pmaximum_decrease + (pmaximum_increase - pmaximum_decrease) * variance
            )
        pelectricity += pvariance

    return pelectricity_vec


def find_pETS(pETS_2024, pETS_dt, pmaximum_increase, pmaximum_decrease):
    """Create a randomised price trajectory for electricity

    Arguments
    ---------
    pETS_2024 : float
    pETS_dt : float
        Randomisation factor in range [-1, +1].
        -1 means low values preferred.
        +1 high values preferred
    pmaximum_increase : float
    pmaximum_decrease : float

    Notes
    -----
    Works similar to find_pNE(), but pETS_dt is translated into a value between 0.6 and 1.5.
    This sets the strength of the exponential increase of pETS. Hard-coded values can be changed, if desired.
    """
    x = (1.5 - 0.6) / 2 * (pETS_dt + 1) + 0.6
    pETS_vec = [pETS_2024]
    for t in range(1, 27):
        variance = random.betavariate(alpha=1, beta=1)
        pvariance = (
            pmaximum_decrease + (pmaximum_increase - pmaximum_decrease) * variance
        )

        pETS = pETS_2024 * (1 + 0.06 * x) ** t + pvariance
        pETS_vec.append(pETS)

    return pETS_vec


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

    Qbiomass_input: float = 362  # [MW]
    Wpower_output_wait: float = 110  # [MW]
    Qheat_output_wait: float = 287  # [MW]
    Wpower_output_invest: float = 53  # [MW]
    Qheat_output_invest: float = 337  # [MW]

    Operating_hours: float = (
        8760 * 0.7
    )  # [h] rule of thumb of ~70 % operating hours/year.
    CO2capture_rate: float = 0.3  # [tCO2/MWh_biomass]

    CO2captured: float = field(init=False)  # [tCO2/year]
    OPEX_power_plant: float = field(init=False)  # [EUR/year]

    # [tCO2/year], about 5 % of CO2 is leaked across the value chain.
    CO2captured = CO2capture_rate * Qbiomass_input * Operating_hours * (1 - 0.05)
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
    pelectricity=50,  # [EUR/MWh]
    pheat=50,  # [EUR/MWh]
    pNE=30,  # [EUR/tCO2]
    pETS_2024=80,  # [EUR/tCO2]
    pbiomass=25,  # [EUR/MWh]
    pelectricity_dt=0.4,  # -1 to 0.4,   sets likelihood of price increase/reduction #NOTE: CHANGE THIS
    pheat_dt=-0.5,  # -0.5 to 0.7, sets likelihood of price increase/reduction
    pNE_dt=0.3,  # -1 to +1,    sets likelihood of price increase/reduction
    pETS_dt=0,  # -1 to +1,    sets exponential increase in ETS prices
    Discount_rate=0.06,  # [-]
    Learning_rate=0.01,  # [-]
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
):
    """Calculate regret and other metrics for two strategies "wait" or "invest" for a SOW

    Notes
    -----
    First generates price trajectories
    Then computes NPV for wait and invest strategies
    Finally, computes metrics
    """

    plant = BeccsPlant()

    # CALCULATE ENERGY/CO2 PRICES FOR THIS SOW:
    # Helping functions are used to construct pseudo-random energy and CO2 price projections. The
    # probabilities of price increases/decreases are influenced by the _dt parameters.

    pelectricity = find_sell_prices(
        pelectricity,
        pelectricity_dt,
        pfloor=5,
    )
    pheat = find_sell_prices(
        pheat,
        pheat_dt,
        pfloor=48,
    )
    pNE = find_sell_prices(
        pNE,
        pNE_dt,
        pfloor=3,
    )
    pETS = find_pETS(pETS_2024, pETS_dt, pmaximum_increase=40, pmaximum_decrease=-40)
    # It is now possible to calculate NPV values!

    # CALCULATE CASH FLOWS BASED ON BOTH INVESTMENT DECISIONS:
    # (1) calculate NPV for not investing, i.e. Waiting:
    CFvec_wait = []

    NPV_wait = 0
    for t in range(0, 27):
        CF = calculate_cash_flow(t, plant, pelectricity, pheat, pbiomass, wait=True)
        if 2023 + t >= yBIOban:
            CF -= (
                plant.CO2captured * pETS[t]
            )  # yBIOban represents a severe restriction of biomass usage, forcing the utility to pay for emission allowances for CO2 not captured.
        NPV_wait += CF / ((1 + Discount_rate) ** t)
        CFvec_wait.append(CF)
    # Now NPV is known for the Wait strategy!

    # (2) calculate NPV (and IRR) for the Invest strategy:
    NPV_invest = 0
    CFvec = []
    # NOTE: For this analysis, NPV and IRR use different cashflows and are therefore not directly linked.
    CFvec_IRR = []
    # In this vector we save pNE_max, which is the maximum CO2 price offered from any of the different support policy models.
    pNE_supported = []
    # First two years we pay CAPEX, and we do not have revenues from NEs.
    for t in range(0, 2):
        CF = (
            calculate_cash_flow(t, plant, pelectricity, pheat, pbiomass, wait=True)
            - CAPEX / 2
        )

        # IRR only includes incomes/costs generated by the CAPEX, and not the baseline revenues from heat and power.
        CFIRR = -CAPEX / 2
        NPV_invest += CF / ((1 + Discount_rate) ** t)

        CFvec.append(CF)
        CFvec_IRR.append(CFIRR)
        # The first two years, the maximum CO2 price is just the voluntary carbon market (VCM) price pNE(t).
        pNE_supported.append(pNE[t])

    for t in range(2, 27):
        CFenergy = calculate_cash_flow(
            t, plant, pelectricity, pheat, pbiomass, wait=False
        )
        # This is the energy penalty incurred by the CAPEX investment.
        CFenergy_IRR = (
            -(
                (plant.Wpower_output_wait - plant.Wpower_output_invest)
                * pelectricity[t]
                + (plant.Qheat_output_wait - plant.Qheat_output_invest) * pheat[t]
            )
            * plant.Operating_hours
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

        # Reminder: this cashflow for IRR is different than for NPV.
        CFIRR = CFenergy_IRR + CFCO2
        # When calculating IRR of Investing, we sometimes avoid ETS costs.
        if 2023 + t >= yBIOban:
            CFIRR += plant.CO2captured * pETS[t]

        CFvec.append(CF)
        CFvec_IRR.append(CFIRR)
        pNE_supported.append(pNE_max)
    # Now NPV (and IRR) is known for the Invest strategy!

    regret = calculate_regret(NPV_invest, NPV_wait, investment_decision)

    ## CALCULATE OTHER INTERESTING PARAMETERS:
    pelectricity_mean = np.mean(pelectricity)
    pheat_mean = np.mean(pheat)
    pNE_mean = np.mean(pNE_supported)
    IRR = npf.irr(CFvec_IRR)
    if math.isnan(IRR):
        IRR = 0

    # Now the calculation model is done!
    return (pNE_mean, NPV_invest, regret, NPV_wait, IRR, pelectricity_mean, pheat_mean)


def return_model() -> Model:
    ## DEFINE RHODIUM MODEL
    model = Model(BECCS_investment)
    model.parameters = [
        Parameter("investment_decision"),
        Parameter("pNE"),
        Parameter("pNE_dt"),
        Parameter("pelectricity"),
        Parameter("pelectricity_dt"),
        Parameter("pheat"),
        Parameter("pheat_dt"),
        Parameter("pbiomass"),

        Parameter("pETS_2024"),
        Parameter("pETS_dt"),
        Parameter("Discount_rate"),
        Parameter("CAPEX"),
        Parameter("OPEX_fixed"),
        Parameter("OPEX_variable"),
        Parameter("Cost_transportation"),
        Parameter("Cost_storage"),
        Parameter("Learning_rate"),
        
        Parameter("AUCTION"),
        Parameter("yQUOTA"),
        Parameter("yEUint"),
        Parameter("yBIOban"),
        Parameter("yCLAIM"),
    ]

    model.responses = [
        Response("pNE_mean", Response.INFO),
        Response("NPV_invest", Response.MAXIMIZE),
        Response("Regret", Response.MINIMIZE),
        Response("NPV_wait", Response.MAXIMIZE),
        Response("IRR", Response.MAXIMIZE),
        Response("pelectricity_mean", Response.INFO),
        Response("pheat_mean", Response.INFO),
    ]

    model.levers = [RealLever("investment_decision", 0, 1, length=2)]

    # For uncertainties, some are expanded ranges around values found in the literature, and some are assumed. Refer to full article.
    model.uncertainties = [
        UniformUncertainty("pNE", 20, 400),
        UniformUncertainty("pNE_dt", 5, 40),
        UniformUncertainty("pbiomass", 15, 35),
        UniformUncertainty("pelectricity",5,160),
        UniformUncertainty("pelectricity_dt",5,40),
        UniformUncertainty("pheat",50, 150),
        UniformUncertainty("pheat_dt",1,20),

        UniformUncertainty("pETS_2024", 60, 100),
        UniformUncertainty("pETS_dt", -1, 1),
        UniformUncertainty("Discount_rate", 0.04, 0.10),
        UniformUncertainty("CAPEX", 100 * 10**6, 300 * 10**6),
        UniformUncertainty("OPEX_fixed", 18 * 10**6, 22 * 10**6),
        UniformUncertainty("OPEX_variable", 33, 55),
        UniformUncertainty("Cost_transportation", 17, 27),
        UniformUncertainty("Cost_storage", 6, 23),
        UniformUncertainty("Learning_rate", 0.0075, 0.0125),

        UniformUncertainty("AUCTION", 0, 1),
        UniformUncertainty("yQUOTA", 2030, 2050),
        UniformUncertainty("yEUint", 2035, 2050),
        UniformUncertainty("yBIOban", 2030, 2050),
        UniformUncertainty("yCLAIM", 2023, 2050),
    ]
    return model
