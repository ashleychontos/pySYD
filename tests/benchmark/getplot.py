import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# LOAD IN CATALOGS:
header   = ['KIC' ,'range_low', 'range_up' ,  'numax'   ,'sig_numax',  'dnu'   ,'sig_dnu','Length' ,'DetFlag','SrcFlag']
idl_df   = pd.read_csv('wg1sample_2015_syd.txt',skiprows=20,delimiter='|',names=header)
pysyd_df = pd.read_csv('results/global.csv')
idl_df.KIC    = idl_df.KIC.astype(int)
pysyd_df.star = pysyd_df.star.astype(int)

# MERGE MEASUREMENTS FROM SYD & pySYD CATALOGS:
df = pd.merge(idl_df,pysyd_df, left_on = 'KIC', right_on='star', how='inner',suffixes=['_idl', '_py'] )

df = df[df['numax_smooth']<=3200]

def get_vals(var,df):
	''' 
	Given variable (either dnu or numax), extracts the relevant values as measured by 
	pySYD & IDL.

	@input:
		var : str
			variable of interest; either 'dnu' or 'numax' 
		df : pandas.DataFrame
			table that contains measured global asteroseismic frequencies & their errors for both
			dnu & numax

	@output:
		idl : numpy.ndarray
			values as measured by IDL for given variable 
		pysyd : numpy.ndarray
			values as measured by pySYD for given variable 
		xerr : numpy.ndarray
			errors on IDL measurements
		yerr : numpy.ndarray
			errors on pySYD measurements
	'''

	if var=='numax':
		idl,pysyd=df['numax'],df['numax_smooth']
		xerr=df['sig_numax']
		#yerr=df['numax_smooth_err']
	elif var=='dnu':
		idl,pysyd=df['dnu_idl'],df['dnu_py']
		xerr=df['sig_dnu']
		#yerr=df['dnu_err']
		
	idl   = idl.to_numpy()
	pysyd = pysyd.to_numpy()
	xerr  = xerr.to_numpy()
	# yerr  = yerr.to_numpy()

	return idl,pysyd,xerr#,yerr

def get_axis_labels(var):
	'''Set x & y labels for given subplots.'''
	if var=='numax':
		ylabel=r'$\textrm{pySYD}$' + ' ' +r'$\nu_{\textrm{max}}$'
		xlabel=r'$\textrm{SYD}$' + ' ' + r'$\nu_{\textrm{max}}$'
		name='numax_SYD_vs_pySYD_0404.png'
	elif var =='dnu':
		ylabel=r'$\textrm{pySYD}$' + ' ' +r'$\Delta\nu$'
		xlabel=r'$\textrm{SYD}$' + ' ' +r'$\Delta\nu$'
		name='dnu_SYD_vs_pySYD_0404.png'
	return xlabel,ylabel,name

def set_plot_params(var,ax1,ax2):
	'''Set plot parameters for given subplots.'''
	if var=='dnu':
		ax2.axhline(0.01,c='lightgrey',ls='--')
		ax2.axhline(-0.01,c='lightgrey',ls='--')
	elif var=='numax':
		ax2.axhline(0.01,c='lightgrey',ls='--')
		ax2.axhline(-0.01,c='lightgrey',ls='--')
	mjlength=5
	mnlength=3
	ax1.tick_params(which='both', # Options for both major and minor ticks
	                top=False, # turn off top ticks
	                left=True, # turn off left ticks
	                right=False,  # turn off right ticks
	                bottom=True)# turn off bottom ticks)
	ax1.tick_params(which='minor',length=mnlength) 
	ax2.tick_params(which='minor',axis='y',length=mnlength) 
	plt.minorticks_on()
	ax2.tick_params(axis='x', which='minor', bottom=False)

def get_stats(idl,pysyd):
	'''Given SYD & pySYD measurements, find offset and scatter.'''
	from astropy.stats import mad_std

	residual = (idl-pysyd)/idl
	
	STR1='Offset: {0:.5f}'.format(np.median(residual))+' $\pm$ ' +'{0:.5f}'.format(1.25*np.std(residual)/np.sqrt(len(residual)))
	
	STR2='Scatter (MAD): {0:.5f}'.format(mad_std(residual))
	STR3='Scatter (SD): {0:.5f}'.format(np.std(residual))
	STR=STR1+'\n'+STR2+'\n'+STR3
	return STR

def set_axis_lims(var,ax1,ax2):
	'''Set x & y limits for each subplot.'''
	if var=='dnu':
		ax2.set_ylim(-0.012,0.012)
	elif var=='numax':
		ax2.set_ylim(-0.018,0.012)

def create_plot(var,show=False,save=True):
	'''
	Creates a comparison plot for either dnu or numax with residuals.

	@input:
		var : str
			variable for which to create the comparison plot; either 'dnu' or 'numax' 
		show : boolean
			to show the plot after creation; default is True 
		save : boolean
			to save the plot after creation; default is False 

	'''
	xlabel,ylabel,name =get_axis_labels(var)
	idl,pysyd,xerr=get_vals(var,df)

	plt.figure(figsize=(6,6))
	plt.rc('text', usetex=True)

	gs = gridspec.GridSpec(8, 6,hspace=0)  
	ax1 = plt.subplot(gs[0:6, 0:6])
	ax1.plot(idl,idl,c='k',ls='--')
	ax1.errorbar(idl,pysyd,fmt='o',mec='k',mfc='none',color='lightgrey')
	ax1.set_ylabel(ylabel)

	ax2 = plt.subplot(gs[6:8, 0:6])
	ax2.errorbar(idl,(idl-pysyd)/idl,fmt='o',mec='k',mfc='none',color='lightgrey')
	ax2.axhline(0,c='k',ls='--')
	ax2.set_ylabel(r'$\frac{\textrm{SYD -- pySYD}}{\textrm{SYD}}$',fontsize=15)
	ax2.set_xlabel(xlabel)
	set_axis_lims(var,ax1,ax2)
	
	# ADD PERFORMANCE STATS TO THE PLOT:
	STR=get_stats(idl,pysyd)
	t=ax1.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='white',edgecolor='k'))

	# CHANGE PLOT DETAILS
	set_plot_params(var,ax1,ax2)
	plt.tight_layout()
	
	if save:
		plt.savefig(name,dpi=500)
	if show: plt.show()

create_plot('numax')
create_plot('dnu')