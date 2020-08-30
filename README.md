# SYD
Translation of the asteroseismic SYD pipeline ([Huber et al. 2009](https://ui.adsabs.harvard.edu/abs/2009CoAst.160...74H/abstract)) from IDL into python.

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
- QUAKES.py : asteroseismology tools created in parallel to SYD

Documentation and code under construction.
