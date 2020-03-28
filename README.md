# SYD
Translating the asteroseismic pipeline SYD from IDL into python with the intent to release the first public, non-expert, user-friendly asteroseismic pipeline to extract global seismic parameters.

SYD works in two major parts:
1) findex : crudely although automatically finding the excess before fitting for the background
2) fitbg : background fit and subtraction

#### SYD/Files
All files required to run SYD reside in SYD/Files. For a basic first iteration, the params_findex.txt and the params_fitbg.txt do not need to be changed. The todo.txt file is a basic list of all stars that SYD will run in a single go. Unique identifiers are required, although specifying TIC/KIC/EPIC is not necessary.

SYD/Files/todo requires both the light curve and power spectrum text file formatted in such a way that ID_LC.txt and ID_PS.txt, where ID exactly matches that in the todo.txt file one directory up. The output will be in this same repo within a results/ folder.

#### SYD/Code
- utils.py : bulk of the SYD pipeline
- functions.py : models, distributions, ffts, smoothing functions are all in this file
- SYD.py : initiates SYD through utils.main
- QUAKES.py : asteroseismology pipeline created in parallel with SYD under advisor Dan Huber (i.e. a more pythonic approach to asteroseismology). The initial use and intent of this pipeline also considers planet hosts and has additional time domain tools to remove planet transits and other anomalies in the time domain.

## Example: alpha Mensae (TIC 141810080)

Below is an example of the output 
