# SYD
Translating the asteroseismic pipeline SYD from IDL into python with the intent to release the first public, non-expert, user-friendly asteroseismic pipeline to extract global seismic parameters.

#### SYD/Files
All files required to run SYD reside in SYD/Files. For a basic first iteration, the params_findex.txt and the params_fitbg.txt do not need to be changed. The todo.txt file is a basic list of all stars that this will be run on in a single go. Unique identifiers is helpful, although specifying TIC/KIC/EPIC is not necessary.

SYD/Files/todo requires both the light curve and power spectrum text file formatted in such a way that ID_LC.txt and ID_PS.txt. The output will be in this same repo within a results/ folder.

### Example: alpha Mensae (TIC 141810080)

