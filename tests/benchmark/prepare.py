import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io import ascii

fits_dir   = 'fits/'							# location of downloaded .FITS files
txt_dir    = 'data/'							# location to save .txt files
fits_files = glob.glob(fits_dir+'*.fits')

for file in fits_files:
	with fits.open(file) as hdul:
		hdr   = hdul[0].header
		data  = hdul[1].data
		freq  = data['FREQUENCY']
		psd   = data['PSD']
		KICID = hdr['KEPLERID'] 

		ascii.write([freq,psd], txt_dir+'%s_PS.txt'%KICID, Writer=ascii.NoHeader, overwrite=True)