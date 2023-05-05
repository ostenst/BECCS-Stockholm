# Questions from Will

- What is the start year of the analysis? Why are both 2023 and 2024 mentioned in the code?
- CO2 Capture rate and operating hours could be subject to uncertainty (operation could be harder than expected, biomass quality may be poor etc.)
- Is there any statistical relationship (e.g. correlation) between heat, electricity and emissions prices? To what extent are/not these represented in the prices?
- Why is price of biomass held constant?
- What justification for the synthetic price functions?
- Why set trajectories for `pelectricity_dt` and `pheat_dt` instead of including them in the uncertainties and using them as a means for exploring the scenario space?

# Installation

Install a custom version of Rhodium from @ostenst's fork:

    pip install git+https://github.com/ostenst/Rhodium@nodelist#egg=Rhodium
