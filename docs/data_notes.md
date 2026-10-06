# Data notes

## Source and licence
Kelmarsh wind farm data, Zenodo record 5841834 (version 0.0.3).
Author: Charlie Plumley, Cubico Sustainable Investments Ltd.
Licence: CC-BY-4.0 (attribution required).
DOI: https://doi.org/10.5281/zenodo.5841834
Six Senvion MM92 turbines, 10-minute SCADA and status logs, 2016 to mid-2021.

## What I downloaded so far
- SCADA zips for 2016, 2017, 2018, 2019 (status files and turbine 1 data checked)
- Signal mapping and static turbine files

## Format
- Turbine_Data files: 9 comment lines, then header; missing values are NaN; time zone UTC
- Status files: start, end, status type, code, message, IEC category
- Some status events have no end time (many are Informational)

## Signal coverage
- 2016: all temperature and speed signals empty until May (not usable for a full year)
- 2017: about 93 to 100 percent filled across the year
- 2018: about 97 percent; October 69 percent and November 94 percent
- 2019: about 99.8 percent for generator bearings; shaft bearings drop to 47 percent in May

## Planned targets
- Gear oil temperature (gearbox)
- Generator bearing front and rear temperature (generator)
Inputs: wind speed, power, rotor and gearbox speed, nacelle ambient temperature.

## Planned split by time
Train 2017, validate 2018, test 2019 (later years if needed).

## Fault labels: limits
- No gearbox or generator bearing failure found in 2016 to 2019 status logs
- Candidate drive-train outages: converter errors (T6 Nov 2017 110 h; T2 Mar 2018; T3 Apr 2018; T2 Dec 2018), T5 fan oil cooler gear overload Apr 2018
- Exclude events hitting several turbines at once (site-wide, likely grid): 25 Jan 2017, 2 to 3 Nov 2018, 4 to 5 Nov 2019
- Exclude planned service (generator brushes, 25 Mar 2019)
- Few labelled faults, so days of warning will be case studies, not statistics