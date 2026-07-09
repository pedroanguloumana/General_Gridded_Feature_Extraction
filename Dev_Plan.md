# Purpose
The purpose of this package is to generate reusable python code that can be used to identify clusters in gridded climate/meteorological data. The data can be imagined as gridded precipitation data, but can also be though of as OLR, brightness temperature, or other field. 

# Design choice
The code should be written in a reusable way. That is, it should be pip-installed (but hosted on my github) and usable across different HPC environments. That is, I can import feature-extraction at the top of my script and use the code in a nice way. 

The end point of this code should be a csv file where every row is a feature, and every column a feature statistic to compute (e.g size, total precip, etc). The way each statistic is computed should be given by the used. 
In the case of GPM level 2 data, or sattelite data with a swath, whether a given feature intersects the boundary (and thus may extend beyond the observable zone) should be recorded as well. 

To emulate the effects of e.g. GPM's swath, there should be an option that, if enabled, segments the data into strips of some width and equatorial inclination angle to record if a feature falls within such a hypothetical swath. It is ok if the file (in this case assumed like a global model output we want to compare to observations) is cut up into swaths like this and then the segmentation is done; that is, it is not necessary to check _if there exists a swath where a given feature fits_, but is enough to make a bunch of swaths and see what features are within them. 

# User settings (input)
- Files to be processed
- Minimum feature size to be recorded
- What functions to be computed
    - This setting should be structured as a dictionary item with {'column_name': function to run that returns a number}
- Whether to use artificial satellite swath
- Output save path

# Misc
- The provenance of each feature identified should be clear: what file it came from and some id. 
- The parallelism for actual processing will be handled by other bits of code. This code should just focus on the reusable logic. 
- To give some concrete motivation, the first application of this code will be on netcdf files, where precipitation features from GPM level 2 data will be compared to precipitation features from DYAMOND cloud-resolving models. 