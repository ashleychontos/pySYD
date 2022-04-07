.. module:: pysyd

*****************
``pySYD`` library
*****************

Thanks for stopping by the ``pySYD`` documentation and taking an interest in
learning more about how it all works -- we are so *thrilled* to share asteroseismology 
with you!

-----

TL;DR
#####

If you (understandably) do not have time to go through the entire user guide, we have summarized 
a couple important tidbits that we think you should know before using the software.

 - The first is that the userbase for the initial `pySYD` release was intended for non-expert 
   astronomers. **With this in mind, the software was originally developed to be as hands-off as
   possible -- as a *strictly* command-line end-to-end tool.** However since then, the software has 
   become more modular in recent updates, thus enabling broader capabilities that can be used across 
   other applications (e.g., Jupyter notebooks). 
 - In addition to being a command-line tool, the software is optimized for running many stars. 
   This means that many of the options that one would typically use or prefer, such as printing 
   output information and displaying figures, is `False` by default. For our purposes 
   here though, we will invoke them to better understand how the software operates. 

-----

.. toctree::
   :titlesonly:
   :maxdepth: 2
   :caption: User Guide

   library/about
   library/input
   library/pipeline
   library/target	      
   library/utils
   library/output

-----

